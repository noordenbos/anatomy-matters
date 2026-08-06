"""Validation cohort: incremental OS prognostic value of molecular classifiers.

Primary: adjusted incremental likelihood-ratio χ² (classifier added to clinical baseline).
Secondary: bootstrap ΔC-index (full vs baseline Cox).
Tertiary: ΔAIC / ΔBIC and interpretability metadata.

Baseline is ``Sex + Location`` by default; optional covariates (Age, Ann Arbor, IPI)
are included automatically when present in the validation metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.exceptions import ConvergenceError, ConvergenceWarning
from scipy.stats import chi2

from .integration_figures import (
    METRIC_FEATURE_ORDER,
    _SKIP_FEATURE_VALUES,
    _prepare_classifier_series,
    _save_show_close,
    benjamini_hochberg,
)
from .validation_cox import (
    ENDPOINT,
    EVENT_COL,
    PENALIZER,
    TIME_COL,
    prepare_cox_survival,
)

CONTINUOUS_BASELINE = frozenset({"Age"})

BASELINE_CORE: tuple[str, ...] = ("Sex", "Location")

UNCLASSIFIED_CLASSIFIER_LEVEL = "unclassified"
RARE_LEVEL_MIN_N = 5
RARE_LEVEL_MIN_EVENTS = 2
RARE_COLLAPSED_LEVEL = "rare/unstable"

# First matching column in ``pred`` / meta is used when available.
OPTIONAL_BASELINE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "Age": ("age", "Age", "age_years"),
    "Ann_Arbor": ("ann_arbor_stage", "Ann Arbor stage:", "ann_arbor", "stage"),
    "IPI_score": ("ipi_score", "IPI", "ipi", "IPI/IELSG-score:"),
}


@dataclass(frozen=True)
class ClassifierSpec:
    column: str
    label: str
    panel: str  # "gep" | "genomic"
    reference: str | None = None


CLASSIFIER_SPECS: tuple[ClassifierSpec, ...] = (
    ClassifierSpec("tumorimmune_archetype", "This work (immune archetype)", "gep", "complex immune"),
    ClassifierSpec("lymphomap", "Li 2025 LymphoMAP", "gep", "LN"),
    ClassifierSpec("Ciav_Cluster", "Ciavarella 2018 Cluster", "gep", "Cold"),
    ClassifierSpec("KotlovSig", "Kotlov 2021 LME", "gep", "Depleted"),
    ClassifierSpec("COO_NanoString", "Cell of Origin", "gep", "GCB"),
    ClassifierSpec("Lymphoma_Ecotype_confident", "Steen 2021 EcoTyper (confident)", "gep", "LE1"),
    ClassifierSpec("Lymphoma_Ecotype", "Steen 2021 EcoTyper (best match)", "gep", "LE1"),
    ClassifierSpec("Lymphgen", "Wright 2020 Lymphgen", "genomic", "BN2"),
    ClassifierSpec("DLBclass", "Chapuy 2025 DLBclass", "genomic", "C1"),
    ClassifierSpec("HMRN", "Lacy 2020 HMRN", "genomic", "C1"),
    ClassifierSpec("LymphPlex", "Shen 2023 LymphPlex", "genomic", "BN2"),
)


PRESPECIFIED_CONTRASTS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "This work (immune archetype)": (
        ("CP vs rest", ("cytotoxic predominant",)),
        ("Low immune vs rest", ("low immune",)),
        ("Complex immune vs rest", ("complex immune",)),
    ),
    "Li 2025 LymphoMAP": (
        ("TEX vs rest", ("TEX",)),
        ("FMAC vs rest", ("FMAC",)),
        ("LN vs rest", ("LN",)),
    ),
    "Ciavarella 2018 Cluster": (
        ("Hot vs rest", ("Hot",)),
        ("Cold vs rest", ("Cold",)),
    ),
    "Kotlov 2021 LME": (
        ("Depleted vs rest", ("Depleted",)),
        ("Mesenchymal vs rest", ("Mesenchymal",)),
        ("Inflammatory vs rest", ("Inflammatory",)),
    ),
    "Cell of Origin": (
        ("ABC vs rest", ("ABC",)),
        ("GCB vs rest", ("GCB",)),
    ),
    "Steen 2021 EcoTyper (confident)": (
        ("LE6 vs rest", ("LE6",)),
        ("LE9 vs rest", ("LE9",)),
        ("LE1 vs rest", ("LE1",)),
    ),
    "Steen 2021 EcoTyper (best match)": (
        ("LE6 vs rest", ("LE6",)),
        ("LE9 vs rest", ("LE9",)),
        ("LE1 vs rest", ("LE1",)),
    ),
    "Wright 2020 Lymphgen": (
        ("MCD vs rest", ("MCD",)),
        ("EZB vs rest", ("EZB",)),
        ("BN2 vs rest", ("BN2",)),
    ),
    "Chapuy 2025 DLBclass": (
        ("C5 vs rest", ("C5",)),
        ("C1 vs rest", ("C1",)),
        ("C3/C4 vs rest", ("C3", "C4")),
    ),
    "Lacy 2020 HMRN": (
        ("C5 vs rest", ("C5",)),
        ("C6 vs rest", ("C6",)),
        ("C1 vs rest", ("C1",)),
    ),
    "Shen 2023 LymphPlex": (
        ("MCD vs rest", ("MCD",)),
        ("TP53 vs rest", ("TP53",)),
        ("EZB vs rest", ("EZB",)),
    ),
}


def _first_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


_PCNSL_LOCATION_LABELS = frozenset({"cns", "pcns", "pcnsl"})


def _is_pcnsl_location(value: object) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() in _PCNSL_LOCATION_LABELS


def _coerce_ipi_baseline(
    ipi: pd.Series,
    *,
    location: pd.Series | None = None,
) -> pd.Series:
    """Deprecated shim: prefer ``validation_ipi.assign_ipi_ielsg_buckets``.

    Kept for callers that only have a raw score series. Maps to primary buckets
    ``0-2`` vs ``>=3``. Does **not** force PCNSL to high risk.
    """
    from .validation_ipi import BUCKET_PRIMARY_HIGH, BUCKET_PRIMARY_LOW

    out = pd.Series(pd.NA, index=ipi.index, dtype=object)
    for idx, value in ipi.items():
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        text = str(value).strip().lower()
        if text == "" or text in {"nan", "none", "na", "unk", "unknown", "<na>"}:
            continue
        if "low-risk" in text or text in {"0", "1", "0-1", "0-2"}:
            out[idx] = BUCKET_PRIMARY_LOW
            continue
        if text in {"3", "4", "5", ">=3", ">3"}:
            out[idx] = BUCKET_PRIMARY_HIGH
            continue
        if "high-risk" in text or text in {"2-5"}:
            # Factor string spans both primary buckets (2 vs 3–5) — leave NA.
            continue
        num = pd.to_numeric(value, errors="coerce")
        if pd.notna(num):
            out[idx] = BUCKET_PRIMARY_LOW if float(num) <= 2 else BUCKET_PRIMARY_HIGH
    return out


def _attach_optional_baseline(surv: pd.DataFrame, pred: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add optional clinical covariates to ``surv`` when columns exist in ``pred``."""
    from .validation_ipi import attach_ipi_ielsg_to_survival

    extra: list[str] = []
    out = surv.copy()
    pred = pred.copy()
    pred.index = pred.index.astype(str)
    out.index = out.index.astype(str)

    age_col = _first_column(pred, OPTIONAL_BASELINE_CANDIDATES["Age"])
    if age_col is not None:
        out["Age"] = pd.to_numeric(pred[age_col], errors="coerce").reindex(out.index)
        extra.append("Age")

    stage_col = _first_column(pred, OPTIONAL_BASELINE_CANDIDATES["Ann_Arbor"])
    if stage_col is not None:
        stage = pred[stage_col].astype(str).str.strip().reindex(out.index)
        stage = stage.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        if stage.notna().any():
            out["Ann_Arbor"] = stage
            extra.append("Ann_Arbor")

    out, ipi_extra = attach_ipi_ielsg_to_survival(out, pred, include_secondary=True)
    extra.extend(ipi_extra)

    return out, extra


def build_survival_with_classifiers(
    pred: pd.DataFrame,
    case_classifications: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Merge OS table, optional clinical fields, and classifier labels."""
    surv = prepare_cox_survival(pred)
    cc = case_classifications.copy()
    cc.index = cc.index.astype(str)
    surv.index = surv.index.astype(str)
    overlap = [c for c in cc.columns if c not in surv.columns]
    surv = surv.join(cc[overlap], how="left")
    surv, optional_baseline = _attach_optional_baseline(surv, pred)
    baseline = list(BASELINE_CORE) + optional_baseline
    return surv, baseline


def _classifier_categorical(series: pd.Series, label: str, reference: str | None) -> pd.Categorical:
    present = [str(x) for x in series.dropna().unique()]
    if not present:
        return pd.Categorical(series, categories=[], ordered=False)
    levels: list[str] = []
    if reference and reference in present:
        levels.append(reference)
    for level in METRIC_FEATURE_ORDER.get(label, []):
        if level in present and level not in levels:
            levels.append(level)
    for level in sorted(set(present) - set(levels)):
        levels.append(level)
    if UNCLASSIFIED_CLASSIFIER_LEVEL in levels:
        levels = [lv for lv in levels if lv != UNCLASSIFIED_CLASSIFIER_LEVEL]
        levels.append(UNCLASSIFIED_CLASSIFIER_LEVEL)
    return pd.Categorical(series, categories=levels, ordered=False)


def _prepare_benchmark_classifier_series(series: pd.Series) -> pd.Series:
    """Map missing / unknown classifier labels to a shared unclassified level."""
    out = series.astype(str).str.strip()
    missing = series.isna() | out.str.lower().isin(_SKIP_FEATURE_VALUES)
    return out.where(~missing, UNCLASSIFIED_CLASSIFIER_LEVEL)


def _shared_benchmark_cohort(
    surv: pd.DataFrame,
    baseline: list[str],
    *,
    genomic_only: bool,
) -> pd.Index:
    """Patients with complete OS + baseline (optionally restricted to genomic-tested)."""
    cols = [TIME_COL, EVENT_COL, *baseline]
    idx = surv[cols].replace([np.inf, -np.inf], np.nan).dropna().index.astype(str)
    if not genomic_only:
        return pd.Index(idx)
    if "genomic_tested" not in surv.columns:
        return pd.Index([], dtype=str)
    tested = surv.loc[idx, "genomic_tested"].fillna(False).astype(bool)
    return pd.Index(idx[tested])


def _analysis_cohort(
    surv: pd.DataFrame,
    spec: ClassifierSpec,
    baseline: list[str],
    *,
    cohort_index: pd.Index | None = None,
    benchmark_mode: bool = False,
) -> pd.DataFrame | None:
    if spec.column not in surv.columns:
        return None

    work = surv.copy()
    if cohort_index is not None:
        work = work.loc[work.index.intersection(cohort_index.astype(str))]
    elif spec.panel == "genomic" and "genomic_tested" in work.columns:
        work = work.loc[work["genomic_tested"].fillna(False).astype(bool)]

    if benchmark_mode:
        clf = _prepare_benchmark_classifier_series(work[spec.column])
    else:
        clf = _prepare_classifier_series(work[spec.column], spec.label)
    work["classifier"] = _classifier_categorical(clf, spec.label, spec.reference)

    keep_cols = [TIME_COL, EVENT_COL, *baseline, "classifier"]
    work = work[keep_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if not benchmark_mode:
        work = work.loc[work["classifier"].notna()].copy()

    if len(work) < 20 or int(work[EVENT_COL].sum()) < 5:
        return None
    if work["classifier"].nunique() < 2:
        return None
    return work


def _cox_model_design(surv_df: pd.DataFrame, factor_cols: tuple[str, ...]) -> pd.DataFrame:
    cols = [TIME_COL, EVENT_COL, *factor_cols]
    df = surv_df[cols].copy().replace([np.inf, -np.inf], np.nan)
    categorical = [c for c in factor_cols if c not in CONTINUOUS_BASELINE]
    continuous = [c for c in factor_cols if c in CONTINUOUS_BASELINE]

    if categorical:
        for col in categorical:
            if hasattr(df[col], "cat"):
                df[col] = pd.Categorical(df[col], categories=df[col].cat.categories, ordered=False)
        df = pd.get_dummies(df, columns=categorical, drop_first=True)

    for col in continuous:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna()


def _aic_bic(cph: CoxPHFitter, n: int) -> tuple[float, float]:
    k = len(cph.params_)
    ll = float(cph.log_likelihood_)
    aic = -2.0 * ll + 2.0 * k
    bic = -2.0 * ll + k * np.log(max(n, 1))
    return aic, bic


def _cox_concordance(cph: CoxPHFitter, df: pd.DataFrame) -> float:
    risk = cph.predict_partial_hazard(df)
    return float(concordance_index(df[TIME_COL].values, -risk.values, df[EVENT_COL].values))


def _fit_cox_local(df: pd.DataFrame, *, penalizer: float = PENALIZER) -> CoxPHFitter:
    cph = CoxPHFitter(penalizer=penalizer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        cph.fit(df, duration_col=TIME_COL, event_col=EVENT_COL)
    if not np.isfinite(float(cph.log_likelihood_)):
        raise ValueError("non-finite Cox log-likelihood")
    return cph


def _fit_incremental_models(
    work: pd.DataFrame,
    baseline: list[str],
    *,
    penalizer: float = PENALIZER,
) -> tuple[CoxPHFitter, CoxPHFitter, pd.DataFrame, pd.DataFrame]:
    reduced_factors = tuple(baseline)
    full_factors = tuple([*baseline, "classifier"])
    df_reduced = _cox_model_design(work, reduced_factors)
    df_full = _cox_model_design(work, full_factors)
    if df_reduced.shape[0] < 20 or df_full.shape[0] < 20:
        raise ValueError("insufficient complete cases after design-matrix construction")
    return _fit_cox_local(df_reduced, penalizer=penalizer), _fit_cox_local(df_full, penalizer=penalizer), df_reduced, df_full


def _incremental_lrt_from_models(
    cph_reduced: CoxPHFitter,
    cph_full: CoxPHFitter,
    *,
    n_events: int,
    method: str,
) -> dict[str, object]:
    lr = 2 * (cph_full.log_likelihood_ - cph_reduced.log_likelihood_)
    df_diff = len(cph_full.params_) - len(cph_reduced.params_)
    negative = bool(lr < -1e-8)
    lr_for_p = max(float(lr), 0.0)
    return {
        "lr_statistic": float(lr_for_p),
        "lr_statistic_raw": float(lr),
        "lr_df": int(df_diff),
        "lr_p_value": float(chi2.sf(lr_for_p, df_diff)),
        "lrt_method": method,
        "lrt_negative_raw": negative,
        "events_per_classifier_df": round(n_events / max(df_diff, 1), 2),
    }


def _fit_unpenalized_lrt(
    work: pd.DataFrame,
    baseline: list[str],
    *,
    n_events: int,
) -> tuple[dict[str, object], CoxPHFitter, CoxPHFitter, pd.DataFrame, pd.DataFrame]:
    """Fit unpenalized Cox for classical LRT, falling back to penalized for stability."""
    try:
        cph_r, cph_f, df_r, df_f = _fit_incremental_models(work, baseline, penalizer=0.0)
        lrt = _incremental_lrt_from_models(cph_r, cph_f, n_events=n_events, method="unpenalized")
        if lrt["lrt_negative_raw"]:
            raise ValueError("negative unpenalized nested LRT")
        return lrt, cph_r, cph_f, df_r, df_f
    except (ConvergenceError, ValueError, np.linalg.LinAlgError, FloatingPointError):
        last_error: Exception | None = None
        for penalizer in (PENALIZER, 0.05, 0.1):
            try:
                cph_r, cph_f, df_r, df_f = _fit_incremental_models(work, baseline, penalizer=penalizer)
                lrt = _incremental_lrt_from_models(
                    cph_r,
                    cph_f,
                    n_events=n_events,
                    method=f"penalized_fallback_{penalizer:g}",
                )
                return lrt, cph_r, cph_f, df_r, df_f
            except (ConvergenceError, ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                last_error = exc
        raise ValueError(f"Cox model failed after penalized fallbacks: {last_error}") from last_error


def _classifier_level_qc(
    work: pd.DataFrame,
    *,
    spec: ClassifierSpec,
    analysis_mode: str,
    contrast: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    grouped = work.groupby(work["classifier"].astype(str), observed=False)
    for level, sub in grouped:
        n = int(len(sub))
        events = int(sub[EVENT_COL].sum())
        rows.append(
            {
                "classifier": spec.label,
                "column": spec.column,
                "panel": spec.panel,
                "analysis_mode": analysis_mode,
                "contrast": contrast or "",
                "level": str(level),
                "n": n,
                "events": events,
                "event_rate": events / max(n, 1),
                "is_unclassified": str(level) == UNCLASSIFIED_CLASSIFIER_LEVEL,
                "is_rare": n < RARE_LEVEL_MIN_N or events < RARE_LEVEL_MIN_EVENTS,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df, {
            "min_level_n": np.nan,
            "min_level_events": np.nan,
            "n_rare_levels": np.nan,
            "n_unclassified": 0,
        }
    return df, {
        "min_level_n": int(df["n"].min()),
        "min_level_events": int(df["events"].min()),
        "n_rare_levels": int(df["is_rare"].sum()),
        "n_unclassified": int(df.loc[df["is_unclassified"], "n"].sum()),
    }


def _collapse_rare_classifier_levels(work: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Collapse globally unstable levels before multi-level Cox tests."""
    out = work.copy()
    stats = (
        out.groupby(out["classifier"].astype(str), observed=False)[EVENT_COL]
        .agg(n="size", events="sum")
        .reset_index()
    )
    rare = stats.loc[
        (stats["n"] < RARE_LEVEL_MIN_N) | (stats["events"] < RARE_LEVEL_MIN_EVENTS),
        "classifier",
    ].astype(str)
    rare_levels = [x for x in rare.tolist() if x != UNCLASSIFIED_CLASSIFIER_LEVEL]
    if not rare_levels:
        return out, []
    labels = out["classifier"].astype(str)
    labels = labels.where(~labels.isin(rare_levels), RARE_COLLAPSED_LEVEL)
    out["classifier"] = pd.Categorical(labels)
    collapsed_stats = (
        out.groupby(out["classifier"].astype(str), observed=False)[EVENT_COL]
        .agg(n="size", events="sum")
    )
    if RARE_COLLAPSED_LEVEL in collapsed_stats.index:
        rare_bucket = collapsed_stats.loc[RARE_COLLAPSED_LEVEL]
        if rare_bucket["n"] < RARE_LEVEL_MIN_N or rare_bucket["events"] < RARE_LEVEL_MIN_EVENTS:
            out = out.loc[out["classifier"].astype(str) != RARE_COLLAPSED_LEVEL].copy()
            labels = out["classifier"].astype(str)
            out["classifier"] = pd.Categorical(labels)
            collapsed_stats = (
                out.groupby(out["classifier"].astype(str), observed=False)[EVENT_COL]
                .agg(n="size", events="sum")
            )
    bad = collapsed_stats[(collapsed_stats["n"] < RARE_LEVEL_MIN_N) | (collapsed_stats["events"] < RARE_LEVEL_MIN_EVENTS)]
    if not bad.empty:
        for level in bad.index.astype(str):
            if level in {UNCLASSIFIED_CLASSIFIER_LEVEL, RARE_COLLAPSED_LEVEL}:
                continue
            labels = labels.where(labels != level, RARE_COLLAPSED_LEVEL)
        out["classifier"] = pd.Categorical(labels)
    return out, rare_levels


def _bootstrap_levels_stable(work: pd.DataFrame) -> bool:
    stats = work.groupby(work["classifier"].astype(str), observed=False)[EVENT_COL].agg(n="size", events="sum")
    return bool(((stats["n"] >= RARE_LEVEL_MIN_N) & (stats["events"] >= RARE_LEVEL_MIN_EVENTS)).all())


def bootstrap_delta_cindex(
    work: pd.DataFrame,
    baseline: list[str],
    *,
    n_boot: int = 300,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Median ΔC-index and 95% bootstrap CI over fixed fitted risk scores."""
    if n_boot <= 0:
        return np.nan, np.nan, np.nan
    try:
        cph_r, cph_f, df_r, df_f = _fit_incremental_models(work, baseline, penalizer=PENALIZER)
    except (ConvergenceError, ValueError, np.linalg.LinAlgError, FloatingPointError):
        return np.nan, np.nan, np.nan
    common = df_r.index.intersection(df_f.index)
    if len(common) < 20:
        return np.nan, np.nan, np.nan
    df_r = df_r.loc[common]
    df_f = df_f.loc[common]
    time = df_f[TIME_COL].to_numpy(dtype=float)
    event = df_f[EVENT_COL].to_numpy(dtype=float)
    risk_r = -cph_r.predict_partial_hazard(df_r).to_numpy(dtype=float).reshape(-1)
    risk_f = -cph_f.predict_partial_hazard(df_f).to_numpy(dtype=float).reshape(-1)
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    idx = np.arange(len(common))
    for _ in range(n_boot):
        sample = rng.choice(idx, size=len(idx), replace=True)
        if event[sample].sum() < 5:
            continue
        try:
            c_r = concordance_index(time[sample], risk_r[sample], event[sample])
            c_f = concordance_index(time[sample], risk_f[sample], event[sample])
            deltas.append(float(c_f - c_r))
        except (ZeroDivisionError, ValueError):
            continue
    if not deltas:
        return np.nan, np.nan, np.nan
    arr = np.asarray(deltas, dtype=float)
    return float(np.median(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def evaluate_classifier(
    surv: pd.DataFrame,
    spec: ClassifierSpec,
    baseline: list[str],
    *,
    cohort_index: pd.Index | None = None,
    benchmark_mode: bool = False,
    n_boot: int = 300,
    seed: int = 0,
    analysis_mode: str | None = None,
) -> dict:
    """Incremental OS metrics for one classifier vs the shared baseline."""
    analysis_mode = analysis_mode or ("intention_to_classify" if benchmark_mode else "callable_only")
    row: dict = {
        "endpoint": ENDPOINT,
        "classifier": spec.label,
        "column": spec.column,
        "panel": spec.panel,
        "analysis_type": "global",
        "analysis_mode": analysis_mode,
        "contrast": "",
        "status": "ok",
        "skip_reason": "",
        "baseline_covariates": ", ".join(baseline),
    }

    work = _analysis_cohort(
        surv,
        spec,
        baseline,
        cohort_index=cohort_index,
        benchmark_mode=benchmark_mode,
    )
    if work is None:
        if spec.column not in surv.columns:
            row["status"] = "skipped"
            row["skip_reason"] = "column_missing"
        elif spec.panel == "genomic" and "genomic_tested" in surv.columns and not surv["genomic_tested"].fillna(False).any():
            row["status"] = "skipped"
            row["skip_reason"] = "no_genomic_patients"
        else:
            row["status"] = "skipped"
            row["skip_reason"] = "insufficient_data"
        return row

    raw_level_df, raw_level_qc = _classifier_level_qc(work, spec=spec, analysis_mode=f"{analysis_mode}_raw")
    collapsed_levels: list[str] = []
    if work["classifier"].nunique() > 2:
        work, collapsed_levels = _collapse_rare_classifier_levels(work)
        if work["classifier"].nunique() < 2:
            row["status"] = "skipped"
            row["skip_reason"] = "fewer_than_two_levels_after_rare_collapse"
            return row

    n_patients = len(work)
    n_events = int(work[EVENT_COL].sum())
    n_levels = int(work["classifier"].nunique())
    level_df, level_qc = _classifier_level_qc(work, spec=spec, analysis_mode=analysis_mode)
    if benchmark_mode:
        n_unclassified = int((work["classifier"].astype(str) == UNCLASSIFIED_CLASSIFIER_LEVEL).sum())
        pct_missing = float(n_unclassified / max(n_patients, 1))
    else:
        pct_missing = float(1.0 - n_patients / max(len(surv), 1))

    row.update(
        n_patients=n_patients,
        n_events=n_events,
        n_levels=n_levels,
        pct_missing_classifier=round(pct_missing, 4),
        n_raw_levels=int(raw_level_df["level"].nunique()) if not raw_level_df.empty else np.nan,
        n_raw_rare_levels=raw_level_qc["n_rare_levels"],
        n_collapsed_rare_levels=len(collapsed_levels),
        collapsed_rare_levels=";".join(collapsed_levels),
        **level_qc,
    )
    if "Lymphoma_Ecotype_imputed" in surv.columns and spec.column == "Lymphoma_Ecotype":
        imputed = surv.reindex(work.index)["Lymphoma_Ecotype_imputed"].fillna(False).astype(bool)
        row["n_ecotype_imputed"] = int(imputed.sum())
        row["pct_ecotype_imputed"] = round(float(imputed.mean()), 4)

    try:
        lrt, cph_r, cph_f, df_r, df_f = _fit_unpenalized_lrt(work, baseline, n_events=n_events)
    except ValueError as exc:
        row["status"] = "skipped"
        row["skip_reason"] = str(exc)
        return row

    row.update(
        **lrt,
        k_baseline=len(cph_r.params_),
        k_full=len(cph_f.params_),
    )
    row["epv_full"] = round(n_events / max(len(cph_f.params_), 1), 2)
    row["lr_chi2_per_df"] = row["lr_statistic"] / max(row["lr_df"], 1)

    aic_r, bic_r = _aic_bic(cph_r, n_patients)
    aic_f, bic_f = _aic_bic(cph_f, n_patients)
    cindex_r = _cox_concordance(cph_r, df_r)
    cindex_f = _cox_concordance(cph_f, df_f)
    row.update(
        cindex_baseline=cindex_r,
        cindex_full=cindex_f,
        delta_cindex_insample=cindex_f - cindex_r,
        aic_baseline=aic_r,
        aic_full=aic_f,
        delta_aic=aic_f - aic_r,
        bic_baseline=bic_r,
        bic_full=bic_f,
        delta_bic=bic_f - bic_r,
    )

    med, lo, hi = bootstrap_delta_cindex(work, baseline, n_boot=n_boot, seed=seed)
    row.update(
        delta_cindex_boot_median=med,
        delta_cindex_boot_ci_lower=lo,
        delta_cindex_boot_ci_upper=hi,
    )
    return row


def evaluate_prespecified_contrast(
    surv: pd.DataFrame,
    spec: ClassifierSpec,
    baseline: list[str],
    contrast_label: str,
    positive_levels: tuple[str, ...],
    *,
    cohort_index: pd.Index | None = None,
    n_boot: int = 300,
    seed: int = 0,
) -> dict:
    """Adjusted binary Cox/LRT for a prespecified classifier level vs all other callable levels."""
    row: dict = {
        "endpoint": ENDPOINT,
        "classifier": spec.label,
        "column": spec.column,
        "panel": spec.panel,
        "analysis_type": "prespecified_contrast",
        "analysis_mode": "callable_only",
        "contrast": contrast_label,
        "contrast_positive_levels": ";".join(positive_levels),
        "status": "ok",
        "skip_reason": "",
        "baseline_covariates": ", ".join(baseline),
    }
    work = _analysis_cohort(
        surv,
        spec,
        baseline,
        cohort_index=cohort_index,
        benchmark_mode=False,
    )
    if work is None:
        row["status"] = "skipped"
        row["skip_reason"] = "insufficient_callable_data"
        return row
    labels = work["classifier"].astype(str)
    positive = labels.isin(positive_levels)
    if positive.sum() < RARE_LEVEL_MIN_N or work.loc[positive, EVENT_COL].sum() < RARE_LEVEL_MIN_EVENTS:
        row["status"] = "skipped"
        row["skip_reason"] = "positive_level_too_rare"
        row["positive_n"] = int(positive.sum())
        row["positive_events"] = int(work.loc[positive, EVENT_COL].sum())
        return row
    if (~positive).sum() < RARE_LEVEL_MIN_N or work.loc[~positive, EVENT_COL].sum() < RARE_LEVEL_MIN_EVENTS:
        row["status"] = "skipped"
        row["skip_reason"] = "negative_level_too_rare"
        return row
    work = work.copy()
    work["classifier"] = pd.Categorical(
        np.where(positive, contrast_label, "rest"),
        categories=["rest", contrast_label],
        ordered=False,
    )
    n_patients = len(work)
    n_events = int(work[EVENT_COL].sum())
    _, level_qc = _classifier_level_qc(
        work,
        spec=spec,
        analysis_mode="callable_only",
        contrast=contrast_label,
    )
    row.update(
        n_patients=n_patients,
        n_events=n_events,
        n_levels=2,
        pct_missing_classifier=round(float(1.0 - n_patients / max(len(surv), 1)), 4),
        positive_n=int(positive.sum()),
        positive_events=int(work.loc[work["classifier"].astype(str) == contrast_label, EVENT_COL].sum()),
        **level_qc,
    )
    try:
        lrt, cph_r, cph_f, df_r, df_f = _fit_unpenalized_lrt(work, baseline, n_events=n_events)
    except ValueError as exc:
        row["status"] = "skipped"
        row["skip_reason"] = str(exc)
        return row
    row.update(
        **lrt,
        k_baseline=len(cph_r.params_),
        k_full=len(cph_f.params_),
    )
    row["epv_full"] = round(n_events / max(len(cph_f.params_), 1), 2)
    row["lr_chi2_per_df"] = row["lr_statistic"] / max(row["lr_df"], 1)
    aic_r, bic_r = _aic_bic(cph_r, n_patients)
    aic_f, bic_f = _aic_bic(cph_f, n_patients)
    cindex_r = _cox_concordance(cph_r, df_r)
    cindex_f = _cox_concordance(cph_f, df_f)
    row.update(
        cindex_baseline=cindex_r,
        cindex_full=cindex_f,
        delta_cindex_insample=cindex_f - cindex_r,
        aic_baseline=aic_r,
        aic_full=aic_f,
        delta_aic=aic_f - aic_r,
        bic_baseline=bic_r,
        bic_full=bic_f,
        delta_bic=bic_f - bic_r,
    )
    med, lo, hi = bootstrap_delta_cindex(work, baseline, n_boot=n_boot, seed=seed)
    row.update(
        delta_cindex_boot_median=med,
        delta_cindex_boot_ci_lower=lo,
        delta_cindex_boot_ci_upper=hi,
    )
    return row


def run_classifier_os_benchmark(
    pred: pd.DataFrame,
    case_classifications: pd.DataFrame,
    out_dir: Path,
    *,
    n_boot: int = 300,
    seed: int = 0,
    show: bool = True,
) -> pd.DataFrame:
    """Run incremental OS benchmark for all configured classifiers."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    surv, baseline = build_survival_with_classifiers(pred, case_classifications)
    surv.to_csv(out_dir / "classifier_os_survival_table.csv")

    gep_cohort = _shared_benchmark_cohort(surv, baseline, genomic_only=False)
    genomic_cohort = _shared_benchmark_cohort(surv, baseline, genomic_only=True)

    primary_rows = []
    sensitivity_rows = []
    contrast_rows = []
    level_rows = []
    for spec in CLASSIFIER_SPECS:
        cohort = genomic_cohort if spec.panel == "genomic" else gep_cohort
        primary = evaluate_classifier(
            surv,
            spec,
            baseline,
            cohort_index=cohort,
            benchmark_mode=False,
            n_boot=n_boot,
            seed=seed,
            analysis_mode="callable_only",
        )
        primary_rows.append(primary)
        sensitivity_rows.append(
            evaluate_classifier(
                surv,
                spec,
                baseline,
                cohort_index=cohort,
                benchmark_mode=True,
                n_boot=n_boot,
                seed=seed,
                analysis_mode="intention_to_classify",
            )
        )
        work_callable = _analysis_cohort(
            surv,
            spec,
            baseline,
            cohort_index=cohort,
            benchmark_mode=False,
        )
        if work_callable is not None:
            levels, _ = _classifier_level_qc(work_callable, spec=spec, analysis_mode="callable_only")
            level_rows.append(levels)
        work_itc = _analysis_cohort(
            surv,
            spec,
            baseline,
            cohort_index=cohort,
            benchmark_mode=True,
        )
        if work_itc is not None:
            levels, _ = _classifier_level_qc(work_itc, spec=spec, analysis_mode="intention_to_classify")
            level_rows.append(levels)
        for contrast_label, positive in PRESPECIFIED_CONTRASTS.get(spec.label, ()):
            contrast_rows.append(
                evaluate_prespecified_contrast(
                    surv,
                    spec,
                    baseline,
                    contrast_label,
                    positive,
                    cohort_index=cohort,
                    n_boot=n_boot,
                    seed=seed,
                )
            )

    results = pd.DataFrame(primary_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    contrasts = pd.DataFrame(contrast_rows)
    level_qc = pd.concat(level_rows, ignore_index=True) if level_rows else pd.DataFrame()
    if "lr_p_value" not in results.columns:
        results["lr_p_value"] = np.nan
    if "lr_p_value" not in sensitivity.columns:
        sensitivity["lr_p_value"] = np.nan
    if "lr_p_value" not in contrasts.columns:
        contrasts["lr_p_value"] = np.nan

    def add_panel_fdr(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        ok_mask = out["status"] == "ok"
        for panel in ("gep", "genomic"):
            panel_mask = ok_mask & (out["panel"] == panel)
            if panel_mask.any():
                out.loc[panel_mask, "lr_fdr"] = benjamini_hochberg(
                    out.loc[panel_mask, "lr_p_value"].values.astype(float)
                )
        out.loc[~ok_mask, "lr_fdr"] = np.nan
        return out

    results = add_panel_fdr(results)
    sensitivity = add_panel_fdr(sensitivity)
    contrasts = add_panel_fdr(contrasts) if not contrasts.empty else contrasts

    results = results.sort_values(["panel", "lr_p_value"], na_position="last").reset_index(drop=True)
    sensitivity = sensitivity.sort_values(["panel", "lr_p_value"], na_position="last").reset_index(drop=True)
    if not contrasts.empty:
        contrasts = contrasts.sort_values(["panel", "lr_p_value"], na_position="last").reset_index(drop=True)
    results.to_csv(out_dir / "classifier_os_benchmark.csv", index=False)
    sensitivity.to_csv(out_dir / "classifier_os_benchmark_intention_to_classify.csv", index=False)
    contrasts.to_csv(out_dir / "classifier_os_prespecified_contrasts.csv", index=False)
    if not level_qc.empty:
        level_qc.to_csv(out_dir / "classifier_os_level_qc.csv", index=False)
    pd.concat([results, sensitivity, contrasts], ignore_index=True).to_csv(
        out_dir / "classifier_os_benchmark_all_modes.csv",
        index=False,
    )

    plot_incremental_lrt(results, out_dir / "figVal16_incremental_lrt.svg", show=show)
    plot_delta_cindex_bootstrap(results, out_dir / "figVal16_delta_cindex_bootstrap.svg", show=show)
    plot_incremental_lrt(
        sensitivity,
        out_dir / "figVal16_incremental_lrt_intention_to_classify.svg",
        show=show,
    )
    plot_delta_cindex_bootstrap(
        sensitivity,
        out_dir / "figVal16_delta_cindex_bootstrap_intention_to_classify.svg",
        show=show,
    )
    plot_contrast_lrt(contrasts, out_dir / "figVal16_prespecified_contrast_lrt.svg", show=show)
    return results


def plot_incremental_lrt(results: pd.DataFrame, out_svg: Path, *, show: bool = True) -> None:
    """Horizontal bar chart of incremental LRT χ² (primary ranking)."""
    df = results[results["status"] == "ok"].copy()
    if df.empty:
        return
    df = df.sort_values("lr_chi2_per_df" if "lr_chi2_per_df" in df.columns else "lr_statistic")
    y = np.arange(len(df))
    colors = ["#b2182b" if (p < 0.05 if np.isfinite(p) else False) else "#92c5de" for p in df["lr_fdr"]]

    fig, ax = plt.subplots(figsize=(9, max(4.0, 0.45 * len(df) + 1.5)))
    ax.barh(y, df["lr_statistic"], color=colors, edgecolor="0.35", linewidth=0.5)
    ax.set_yticks(y, df["classifier"])
    ax.set_xlabel("Incremental LRT χ² (classifier | baseline)")
    mode = df["analysis_mode"].iloc[0] if "analysis_mode" in df.columns else "global"
    ax.set_title(f"Incremental OS prognostic value — validation ({ENDPOINT}, {mode})")
    for yi, (_, row) in enumerate(df.iterrows()):
        rare = int(row.get("n_rare_levels", 0)) if pd.notna(row.get("n_rare_levels", np.nan)) else 0
        note = (
            f"n={int(row['n_patients'])}, df={int(row['lr_df'])}, "
            f"χ²/df={row.get('lr_chi2_per_df', np.nan):.2g}, "
            f"rare={rare}, FDR={row['lr_fdr']:.3g}"
        )
        ax.text(row["lr_statistic"] + 0.05, yi, note, va="center", fontsize=8, color="0.25")
    fig.tight_layout()
    _save_show_close(fig, out_svg, show=show)


def plot_delta_cindex_bootstrap(results: pd.DataFrame, out_svg: Path, *, show: bool = True) -> None:
    """Bootstrap ΔC-index (full − baseline) with 95% CI."""
    df = results[results["status"] == "ok"].copy()
    if df.empty:
        return
    df = df.sort_values("delta_cindex_boot_median")
    y = np.arange(len(df))
    x = df["delta_cindex_boot_median"].values
    lo = df["delta_cindex_boot_ci_lower"].values
    hi = df["delta_cindex_boot_ci_upper"].values
    err = np.vstack([x - lo, hi - x])

    fig, ax = plt.subplots(figsize=(9, max(4.0, 0.45 * len(df) + 1.5)))
    ax.errorbar(x, y, xerr=err, fmt="o", color="#2166ac", ecolor="0.45", capsize=3, markersize=6)
    ax.axvline(0.0, color="0.35", linestyle="--", linewidth=0.8)
    ax.set_yticks(y, df["classifier"])
    ax.set_xlabel("ΔC-index (bootstrap median, full − baseline)")
    ax.set_title(f"Discrimination gain — validation ({ENDPOINT})")
    fig.tight_layout()
    _save_show_close(fig, out_svg, show=show)


def plot_contrast_lrt(results: pd.DataFrame, out_svg: Path, *, show: bool = True) -> None:
    """Horizontal bar chart of prespecified one-vs-rest contrast LRTs."""
    if results.empty:
        return
    df = results[results["status"] == "ok"].copy()
    if df.empty:
        return
    df["label"] = df["classifier"] + " — " + df["contrast"].astype(str)
    df = df.sort_values("lr_p_value", ascending=False)
    y = np.arange(len(df))
    colors = ["#b2182b" if (p < 0.05 if np.isfinite(p) else False) else "#d1e5f0" for p in df["lr_fdr"]]

    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.35 * len(df) + 1.5)))
    ax.barh(y, df["lr_statistic"], color=colors, edgecolor="0.35", linewidth=0.5)
    ax.set_yticks(y, df["label"])
    ax.set_xlabel("Adjusted one-vs-rest LRT χ²")
    ax.set_title(f"Prespecified classifier contrasts — validation ({ENDPOINT})")
    for yi, (_, row) in enumerate(df.iterrows()):
        note = (
            f"+n={int(row.get('positive_n', 0))}, +ev={int(row.get('positive_events', 0))}, "
            f"p={row['lr_p_value']:.3g}, FDR={row['lr_fdr']:.3g}"
        )
        ax.text(row["lr_statistic"] + 0.05, yi, note, va="center", fontsize=7.5, color="0.25")
    fig.tight_layout()
    _save_show_close(fig, out_svg, show=show)
