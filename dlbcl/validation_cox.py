"""Validation-cohort Cox models mirroring discovery Fig S3A–S3D (nb9 aesthetics).

OS only. Univariable factors are limited to fields present in the validation
workbook (Sex, Location, Archetype, COO). Categorical levels with no patients
after filtering are omitted automatically.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import chi2

from .integration_figures import LOCATION_GROUP_ORDER
from .validation_figures import (
    ARCHETYPE_ORDER,
    CLUSTER_ORDER,
    DISEASE_TYPE_TO_LOCATION,
)

TIME_COL = "time"
EVENT_COL = "event"
ENDPOINT = "OS"

REF_LOCATION = "bone"
REF_ARCHETYPE = "complex immune"
REF_COO = "GCB"

PENALIZER = 0.01

RISK_GRID_HORIZON_YEARS = 2.0
RISK_ALPHA_LO = 0.1
RISK_ALPHA_HI = 1.0
RISK_ALPHA_N_LO = 1
RISK_ALPHA_N_HI = 7

UNIVARIABLE_FACTORS = {
    "Sex": "Sex",
    "Location": "Location",
    "Archetype": "Archetype",
    "COO": "COO",
}

CONTINUOUS_COVARIATES = {"Sex"}

CONTINUOUS_LABELS = {
    "Sex": "male vs female",
}

_LRT_COMPARE_ORDER = [
    "Null → Location",
    "Null → Archetype",
    "Null → Location + Archetype",
    "Location → Location + Archetype",
    "Archetype → Location + Archetype",
]

_PCNSL_LOCATION_LABELS = frozenset({"cns", "pcns", "pcnsl"})
REF_IPI_SCORE = "0-1"

ARCHETYPE_IPI_COHORTS = (
    {
        "key": "exclude_pcnsl",
        "label": "Exclude PCNSL",
        "stem": "cox_multivar_archetype_ipi_exclude_pcnsl_OS",
    },
    {
        "key": "include_pcnsl_ipi_gt3",
        "label": "Non-PCNSL + PCNSL with IPI >3",
        "stem": "cox_multivar_archetype_ipi_include_pcnsl_ipi_gt3_OS",
    },
)


def _set_reference(series: pd.Series, reference: str) -> pd.Categorical:
    levels = [reference] + sorted(x for x in pd.Series(series).dropna().unique() if x != reference)
    return pd.Categorical(series, categories=levels, ordered=False)


def _set_ordered_categorical(series: pd.Series, order: list[str], reference: str) -> pd.Categorical:
    present = set(pd.Series(series).dropna().unique())
    levels = [reference] + [x for x in order if x != reference and x in present]
    levels.extend(sorted(present - set(levels)))
    return pd.Categorical(series, categories=levels, ordered=False)


def prepare_cox_survival(pred: pd.DataFrame) -> pd.DataFrame:
    """Build OS Cox table with discovery-aligned Location / Archetype / COO / Sex."""
    surv = pd.DataFrame(index=pred.index)
    surv[TIME_COL] = pd.to_numeric(pred["follow_up_time"], errors="coerce")
    surv[EVENT_COL] = pd.to_numeric(pred["vital_status"], errors="coerce")
    surv["Location"] = pred["disease_type"].map(DISEASE_TYPE_TO_LOCATION)
    archetype_id = pd.to_numeric(pred["pred_abundance_cluster_30"], errors="coerce")
    surv["Archetype"] = archetype_id.map({1: "low immune", 2: "cytotoxic predominant", 3: "complex immune"})
    sex_map = {"Male": 1.0, "Female": 0.0, "male": 1.0, "female": 0.0}
    surv["Sex"] = pred["sex"].map(sex_map)
    coo_map = {"GCB": "GCB", "Non-GCB": "ABC"}
    surv["COO"] = pred["coo_hans"].map(coo_map)

    surv = surv.dropna(subset=[TIME_COL, EVENT_COL, "Location", "Archetype"])
    surv = surv.loc[surv[TIME_COL] > 0].copy()
    for col in (TIME_COL, EVENT_COL, "Sex"):
        surv[col] = pd.to_numeric(surv[col], errors="coerce")

    if REF_LOCATION in set(surv["Location"].dropna()):
        surv["Location"] = _set_reference(surv["Location"], REF_LOCATION)
    else:
        ref = sorted(surv["Location"].dropna().unique())[0]
        surv["Location"] = _set_reference(surv["Location"], ref)

    if REF_ARCHETYPE in set(surv["Archetype"].dropna()):
        surv["Archetype"] = _set_ordered_categorical(surv["Archetype"], ARCHETYPE_ORDER, REF_ARCHETYPE)
    else:
        ref = sorted(surv["Archetype"].dropna().unique())[0]
        surv["Archetype"] = _set_ordered_categorical(surv["Archetype"], ARCHETYPE_ORDER, ref)

    coo_present = surv["COO"].dropna()
    if len(coo_present) and REF_COO in set(coo_present):
        surv.loc[coo_present.index, "COO"] = _set_reference(coo_present, REF_COO)
    elif len(coo_present):
        ref = sorted(coo_present.unique())[0]
        surv.loc[coo_present.index, "COO"] = _set_reference(coo_present, ref)

    return surv


def _is_pcnsl_location(value: object) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() in _PCNSL_LOCATION_LABELS


def attach_ipi_score(surv: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """Add workbook IPI (0–1 vs >3; PCNSL without score assigned >3) to a Cox table."""
    from .validation_classifier_survival import (
        OPTIONAL_BASELINE_CANDIDATES,
        _coerce_ipi_baseline,
        _first_column,
    )

    out = surv.copy()
    pred = pred.copy()
    pred.index = pred.index.astype(str)
    out.index = out.index.astype(str)

    ipi_col = _first_column(pred, OPTIONAL_BASELINE_CANDIDATES["IPI_score"])
    if ipi_col is None:
        return out

    ipi_mapped = _coerce_ipi_baseline(
        pred[ipi_col].reindex(out.index),
        location=out.get("Location"),
    )
    if ipi_mapped.notna().any():
        out["IPI_score"] = pd.Categorical(
            ipi_mapped,
            categories=["0-1", ">3"],
            ordered=False,
        )
    return out


def prepare_archetype_ipi_survival(pred: pd.DataFrame) -> pd.DataFrame:
    """OS Cox table with Archetype and IPI_score for archetype+IPI multivariable models."""
    surv = prepare_cox_survival(pred)
    return attach_ipi_score(surv, pred)


def cohort_exclude_pcnsl(surv: pd.DataFrame) -> pd.DataFrame:
    """Drop all PCNSL patients."""
    mask = ~surv["Location"].map(_is_pcnsl_location)
    return surv.loc[mask].copy()


def cohort_include_pcnsl_ipi_gt3(surv: pd.DataFrame) -> pd.DataFrame:
    """Keep all non-PCNSL patients plus PCNSL patients with IPI >3."""
    is_pcnsl = surv["Location"].map(_is_pcnsl_location)
    ipi = surv["IPI_score"].astype(str)
    keep = (~is_pcnsl) | (ipi == ">3")
    return surv.loc[keep].copy()


def _archetype_ipi_categorical_spec(surv: pd.DataFrame) -> dict:
    spec = _univariate_categorical_spec(surv)
    if "IPI_score" not in surv.columns:
        return spec
    ipi = surv["IPI_score"].dropna()
    if len(ipi) < 2:
        return spec
    if hasattr(ipi, "cat"):
        levels = list(ipi.cat.categories)
        reference = levels[0] if levels else REF_IPI_SCORE
    else:
        levels = sorted(ipi.unique())
        reference = REF_IPI_SCORE if REF_IPI_SCORE in levels else levels[0]
    spec["IPI_score"] = {
        "col": "IPI_score",
        "reference": reference,
        "levels": levels,
    }
    return spec


def _cohort_filter_fn(key: str):
    if key == "exclude_pcnsl":
        return cohort_exclude_pcnsl
    if key == "include_pcnsl_ipi_gt3":
        return cohort_include_pcnsl_ipi_gt3
    raise ValueError(f"Unknown archetype+IPI cohort key: {key}")


def run_multivariable_archetype_ipi(
    surv: pd.DataFrame,
    *,
    cohort_label: str,
    out_stem: str,
    out_dir: Path,
    show: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Multivariable Cox: Archetype + IPI_score (reference archetype complex immune; IPI 0–1)."""
    out_dir = Path(out_dir)
    work = surv.dropna(subset=[TIME_COL, EVENT_COL, "Archetype", "IPI_score"]).copy()
    if work.shape[0] < 10 or int(work[EVENT_COL].sum()) < 5:
        raise ValueError(f"{cohort_label}: insufficient patients/events for Archetype + IPI Cox")

    categorical_spec = _archetype_ipi_categorical_spec(work)
    mv = _cox_design(work, ("Archetype", "IPI_score"))
    cph = _fit_cox(mv)

    summary = cph.summary.copy()
    summary["HR"] = np.exp(summary["coef"])
    summary["CI_lower"] = np.exp(summary["coef lower 95%"])
    summary["CI_upper"] = np.exp(summary["coef upper 95%"])
    summary.insert(0, "cohort", cohort_label)
    summary.insert(0, "endpoint", ENDPOINT)
    summary.to_csv(out_dir / f"{out_stem}.csv")

    from .cox_forest_plot import save_cox_forest_table

    forest_df = save_cox_forest_table(
        _cox_to_forest_df(
            cph,
            f"Archetype + IPI ({cohort_label})",
            ("Archetype", "IPI_score"),
            categorical_spec,
        ),
        out_dir / f"{out_stem}_forest",
        title=f"Multivariable Cox — {ENDPOINT}: Archetype + IPI ({cohort_label}, validation)",
        log_x=True,
        show=show,
    )
    meta = {
        "n_patients": int(work.shape[0]),
        "n_events": int(work[EVENT_COL].sum()),
    }
    return summary, forest_df, meta


def run_archetype_ipi_cox_models(
    pred: pd.DataFrame,
    out_dir: Path,
    *,
    show: bool = True,
) -> dict[str, object]:
    """Run both validation archetype+IPI multivariable Cox models."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    surv = prepare_archetype_ipi_survival(pred)
    surv.to_csv(out_dir / "cox_archetype_ipi_survival_table.csv")

    results: dict[str, object] = {"surv": surv, "models": {}}
    for spec in ARCHETYPE_IPI_COHORTS:
        filtered = _cohort_filter_fn(spec["key"])(surv)
        summary, forest, meta = run_multivariable_archetype_ipi(
            filtered,
            cohort_label=spec["label"],
            out_stem=spec["stem"],
            out_dir=out_dir,
            show=show,
        )
        results["models"][spec["key"]] = {
            "label": spec["label"],
            "summary": summary,
            "forest": forest,
            **meta,
        }
    return results


def _categorical_levels(surv: pd.DataFrame, col: str) -> list:
    if not hasattr(surv[col], "cat"):
        return sorted(surv[col].dropna().unique())
    return list(surv[col].cat.categories)


def _univariate_categorical_spec(surv: pd.DataFrame) -> dict:
    specs = {}
    if "Location" in surv.columns:
        specs["Location"] = {
            "col": "Location",
            "reference": surv["Location"].cat.categories[0] if hasattr(surv["Location"], "cat") else REF_LOCATION,
            "levels": _categorical_levels(surv, "Location"),
        }
    if "Archetype" in surv.columns:
        specs["Archetype"] = {
            "col": "Archetype",
            "reference": surv["Archetype"].cat.categories[0] if hasattr(surv["Archetype"], "cat") else REF_ARCHETYPE,
            "levels": _categorical_levels(surv, "Archetype"),
        }
    coo = surv["COO"].dropna() if "COO" in surv.columns else pd.Series(dtype=object)
    if len(coo) >= 2:
        specs["COO"] = {
            "col": "COO",
            "reference": coo.cat.categories[0] if hasattr(coo, "cat") else REF_COO,
            "levels": _categorical_levels(surv.loc[coo.index], "COO"),
        }
    return specs


def _design_matrix(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in CONTINUOUS_COVARIATES:
        sub = df[[TIME_COL, EVENT_COL, col]].copy()
        return sub.dropna()
    out = df[[TIME_COL, EVENT_COL, col]].copy().dropna()
    if out[col].nunique() < 2:
        raise ValueError(f"{col} has fewer than two levels")
    dummies = pd.get_dummies(out[col], prefix=col, drop_first=True)
    return pd.concat([out[[TIME_COL, EVENT_COL]], dummies], axis=1)


def _extract_cox_terms(cph: CoxPHFitter, factor_name: str) -> list[dict]:
    rows = []
    for term, row in cph.summary.iterrows():
        rows.append({
            "factor": factor_name,
            "term": term,
            "n_patients": int(cph.event_observed.shape[0]),
            "n_events": int(cph.event_observed.sum()),
            "HR": float(np.exp(row["coef"])),
            "CI_lower": float(np.exp(row["coef lower 95%"])),
            "CI_upper": float(np.exp(row["coef upper 95%"])),
            "coef": float(row["coef"]),
            "p": float(row["p"]),
        })
    return rows


def _expand_univariable_forest(uni_df: pd.DataFrame, categorical_spec: dict) -> pd.DataFrame:
    rows = []
    for factor_label, col in UNIVARIABLE_FACTORS.items():
        if factor_label in CONTINUOUS_LABELS:
            sub = uni_df[uni_df["factor"] == factor_label]
            if sub.empty:
                continue
            r = sub.iloc[0].to_dict()
            r["label"] = CONTINUOUS_LABELS[factor_label]
            r["is_reference"] = False
            rows.append(r)
            continue
        if factor_label not in categorical_spec:
            continue
        spec = categorical_spec[factor_label]
        col_name = spec["col"]
        fac_mask = uni_df["factor"] == factor_label
        n_patients = int(uni_df.loc[fac_mask, "n_patients"].max()) if fac_mask.any() else np.nan
        n_events = int(uni_df.loc[fac_mask, "n_events"].max()) if fac_mask.any() else np.nan
        for level in spec["levels"]:
            term = f"{col_name}_{level}"
            is_ref = level == spec["reference"]
            match = uni_df[(uni_df["factor"] == factor_label) & (uni_df["term"] == term)]
            if is_ref:
                rows.append({
                    "factor": factor_label,
                    "term": term,
                    "label": str(level),
                    "n_patients": n_patients,
                    "n_events": n_events,
                    "HR": 1.0,
                    "CI_lower": 1.0,
                    "CI_upper": 1.0,
                    "coef": 0.0,
                    "p": np.nan,
                    "is_reference": True,
                })
            elif len(match):
                r = match.iloc[0].to_dict()
                r["label"] = str(level)
                r["is_reference"] = False
                rows.append(r)
    return pd.DataFrame(rows)


def _forest_plot_univariable(
    uni_df: pd.DataFrame,
    categorical_spec: dict,
    out_dir: Path,
    *,
    out_stem: str,
    title: str | None = None,
    factors: tuple[str, ...] | None = None,
    show: bool = True,
) -> pd.DataFrame:
    from .cox_forest_plot import save_cox_forest_table

    plot_df = _expand_univariable_forest(uni_df, categorical_spec)
    if factors is not None:
        plot_df = plot_df[plot_df["factor"].isin(factors)].copy()
    return save_cox_forest_table(
        plot_df,
        out_dir / out_stem,
        title=title or f"Univariable Cox — {ENDPOINT} (validation)",
        log_x=False,
        show=show,
    )


def _dummy_term(factor: str, level: str) -> str:
    return f"{factor}_{level}"


def _level_label(factor: str, level: str, *, is_reference: bool = False) -> str:
    return str(level)


def _term_factor_level(term: str, categorical_factors: tuple[str, ...]) -> tuple[str, str]:
    if term in CONTINUOUS_LABELS:
        return term, CONTINUOUS_LABELS[term]
    for factor in sorted(categorical_factors, key=len, reverse=True):
        prefix = f"{factor}_"
        if term.startswith(prefix):
            return factor, term[len(prefix) :]
    return "Model", term.replace("_", " ")


def _cox_to_forest_df(cph: CoxPHFitter, model_label: str, categorical_factors: tuple[str, ...], categorical_spec: dict) -> pd.DataFrame:
    rows = []
    n_patients = int(cph.event_observed.shape[0])
    n_events = int(cph.event_observed.sum())
    for term, row in cph.summary.iterrows():
        rows.append({
            "model": model_label,
            "term": term,
            "label": CONTINUOUS_LABELS.get(term, term.replace("_", " ")),
            "n_patients": n_patients,
            "n_events": n_events,
            "HR": float(np.exp(row["coef"])),
            "CI_lower": float(np.exp(row["coef lower 95%"])),
            "CI_upper": float(np.exp(row["coef upper 95%"])),
            "p": float(row["p"]),
            "is_reference": False,
        })
    forest_df = pd.DataFrame(rows)
    for factor in categorical_factors:
        spec = categorical_spec[factor]
        ref = spec["reference"]
        ref_term = _dummy_term(factor, ref)
        if ref_term not in forest_df["term"].values:
            forest_df = pd.concat([
                forest_df,
                pd.DataFrame([{
                    "model": model_label,
                    "term": ref_term,
                    "label": _level_label(factor, ref, is_reference=True),
                    "n_patients": n_patients,
                    "n_events": n_events,
                    "HR": 1.0,
                    "CI_lower": 1.0,
                    "CI_upper": 1.0,
                    "p": np.nan,
                    "is_reference": True,
                }]),
            ], ignore_index=True)
        for level in spec["levels"]:
            if level == ref:
                continue
            term = _dummy_term(factor, level)
            if term not in forest_df["term"].values:
                forest_df = pd.concat([
                    forest_df,
                    pd.DataFrame([{
                        "model": model_label,
                        "term": term,
                        "label": _level_label(factor, level),
                        "n_patients": n_patients,
                        "n_events": n_events,
                        "HR": np.nan,
                        "CI_lower": np.nan,
                        "CI_upper": np.nan,
                        "p": np.nan,
                        "is_reference": False,
                    }]),
                ], ignore_index=True)
    order = []
    for factor in categorical_factors:
        for level in categorical_spec[factor]["levels"]:
            order.append(_dummy_term(factor, level))
    forest_df["sort_key"] = forest_df["term"].map({t: i for i, t in enumerate(order)}).fillna(999)
    forest_df = forest_df.sort_values("sort_key").drop(columns=["sort_key"])
    factors, labels = zip(
        *[_term_factor_level(term, categorical_factors) for term in forest_df["term"]],
        strict=True,
    )
    forest_df["factor"] = list(factors)
    forest_df["label"] = list(labels)
    return forest_df


def _cox_design(surv_df: pd.DataFrame, factors: tuple[str, ...] = ()) -> pd.DataFrame:
    cols = [TIME_COL, EVENT_COL, *factors]
    df = surv_df[cols].copy().replace([np.inf, -np.inf], np.nan).dropna()
    if not factors:
        return df
    for col in factors:
        if hasattr(df[col], "cat"):
            df[col] = pd.Categorical(df[col], categories=df[col].cat.categories, ordered=False)
    return pd.get_dummies(df, columns=list(factors), drop_first=True)


def _fit_cox(df: pd.DataFrame) -> CoxPHFitter:
    cph = CoxPHFitter(penalizer=PENALIZER)
    cph.fit(df, duration_col=TIME_COL, event_col=EVENT_COL)
    return cph


def _lrt_row(reduced_label: str, full_label: str, cph_reduced: CoxPHFitter, cph_full: CoxPHFitter, n_events: int) -> dict:
    lr = 2 * (cph_full.log_likelihood_ - cph_reduced.log_likelihood_)
    df_diff = len(cph_full.params_) - len(cph_reduced.params_)
    return {
        "endpoint": ENDPOINT,
        "model_reduced": reduced_label,
        "model_full": full_label,
        "n_patients": int(cph_full.event_observed.shape[0]),
        "n_events": int(n_events),
        "lr_statistic": float(lr),
        "df": int(df_diff),
        "p_value": float(chi2.sf(lr, df_diff)),
    }


def _partition_loglik(ll_null: float, ll_loc: float, ll_arch: float, ll_full: float) -> dict:
    total = 2 * (ll_full - ll_null)
    unique_loc = 2 * (ll_full - ll_arch)
    unique_arch = 2 * (ll_full - ll_loc)
    shared_raw = 2 * (ll_loc + ll_arch - ll_full - ll_null)
    shared = max(0.0, float(shared_raw))
    return {
        "endpoint": ENDPOINT,
        "chi2_total": float(total),
        "chi2_unique_location": float(unique_loc),
        "chi2_unique_archetype": float(unique_arch),
        "chi2_shared_raw": float(shared_raw),
        "chi2_shared": shared,
        "shared_suppression": bool(shared_raw < 0),
    }


def _symmetric_lrt_label(row: pd.Series) -> str:
    if row["model_reduced"] == "Null":
        return f"Null → {row['model_full']}"
    return f"{row['model_reduced']} → {row['model_full']}"


def _symmetric_lrt_color(row: pd.Series) -> str:
    if row["model_reduced"] == "Null" and row["model_full"] == "Location":
        return "#377eb8"
    if row["model_reduced"] == "Null" and row["model_full"] == "Archetype":
        return "#e41a1c"
    if row["model_reduced"] == "Null" and row["model_full"] == "Location + Archetype":
        return "#7570b3"
    if row["model_full"] == "Location + Archetype" and row["model_reduced"] == "Location":
        return "#f4a582"
    if row["model_full"] == "Location + Archetype" and row["model_reduced"] == "Archetype":
        return "#92c5de"
    return "0.45"


def _format_lrt_p(p: float) -> str:
    if not np.isfinite(p):
        return "p=NA"
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}"


def _landmark_event_rate(cell_df: pd.DataFrame, horizon: float = RISK_GRID_HORIZON_YEARS) -> tuple[int, int, float]:
    t = pd.to_numeric(cell_df[TIME_COL], errors="coerce")
    ev = pd.to_numeric(cell_df[EVENT_COL], errors="coerce") == 1
    ok = t.notna() & (t > 0)
    t, ev = t[ok], ev[ok]
    if len(t) == 0:
        return 0, 0, np.nan
    events_h = int((ev & (t <= horizon)).sum())
    n_h = int(((t >= horizon) | (ev & (t <= horizon))).sum())
    rate = events_h / n_h if n_h else np.nan
    return events_h, n_h, rate


def _heatmap_cell_alpha(n: int) -> float:
    if n <= 0:
        return 0.0
    if n <= RISK_ALPHA_N_LO:
        return RISK_ALPHA_LO
    if n >= RISK_ALPHA_N_HI:
        return RISK_ALPHA_HI
    frac = (n - RISK_ALPHA_N_LO) / (RISK_ALPHA_N_HI - RISK_ALPHA_N_LO)
    return float(RISK_ALPHA_LO + frac * (RISK_ALPHA_HI - RISK_ALPHA_LO))


def _rgba_heatmap(
    mat: np.ndarray,
    n_mat: np.ndarray,
    *,
    vmin: float = 0,
    vmax: float = 1,
    cmap_name: str = "YlOrRd",
    use_n_alpha: bool = True,
):
    from matplotlib import colormaps
    from matplotlib.colors import Normalize

    cmap_obj = colormaps.get_cmap(cmap_name)
    norm = Normalize(vmin=vmin, vmax=vmax)
    h, w = mat.shape
    rgba = np.ones((h, w, 4))
    rgba[..., 3] = 0.0
    for i in range(h):
        for j in range(w):
            if n_mat[i, j] > 0 and np.isfinite(mat[i, j]):
                r, g, b, _ = cmap_obj(norm(mat[i, j]))
                alpha = _heatmap_cell_alpha(n_mat[i, j]) if use_n_alpha else RISK_ALPHA_HI
                rgba[i, j] = (r, g, b, alpha)
    return rgba, cmap_obj, norm


def run_univariable_cox(surv: pd.DataFrame, out_dir: Path, *, show: bool = True) -> pd.DataFrame:
    categorical_spec = _univariate_categorical_spec(surv)
    uni_rows = []
    for label, col in UNIVARIABLE_FACTORS.items():
        if col not in surv.columns:
            continue
        try:
            df = _design_matrix(surv, col)
        except ValueError:
            continue
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        if df.shape[0] < 10 or int(df[EVENT_COL].sum()) < 5:
            continue
        cph = CoxPHFitter(penalizer=PENALIZER)
        cph.fit(df, duration_col=TIME_COL, event_col=EVENT_COL)
        uni_rows.extend(_extract_cox_terms(cph, label))
    uni_df = pd.DataFrame(uni_rows).sort_values(["factor", "p"])
    uni_df.insert(0, "endpoint", ENDPOINT)
    uni_df.to_csv(out_dir / "cox_univariable_OS.csv", index=False)
    _forest_plot_univariable(
        uni_df, categorical_spec, out_dir,
        out_stem="cox_univariable_forest_OS",
        title=f"Univariable Cox — {ENDPOINT} (validation)",
        show=show,
    )
    _forest_plot_univariable(
        uni_df, categorical_spec, out_dir,
        out_stem="cox_univariable_location_archetype_forest_OS",
        title=f"Univariable Cox — {ENDPOINT}: Location & Archetype (validation)",
        factors=("Location", "Archetype"),
        show=show,
    )
    return uni_df


def run_multivariable_location_archetype(
    surv: pd.DataFrame,
    out_dir: Path,
    *,
    show: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    categorical_spec = _univariate_categorical_spec(surv)
    mv = _cox_design(surv, ("Location", "Archetype"))
    cph = _fit_cox(mv)
    summary = cph.summary.copy()
    summary["HR"] = np.exp(summary["coef"])
    summary["CI_lower"] = np.exp(summary["coef lower 95%"])
    summary["CI_upper"] = np.exp(summary["coef upper 95%"])
    summary.insert(0, "endpoint", ENDPOINT)
    summary.to_csv(out_dir / "cox_multivariable_location_archetype_OS.csv")

    from .cox_forest_plot import save_cox_forest_table

    forest_df = save_cox_forest_table(
        _cox_to_forest_df(
            cph,
            "Location + Archetype",
            ("Location", "Archetype"),
            categorical_spec,
        ),
        out_dir / "cox_multivariable_location_archetype_forest_OS",
        title=f"Multivariable Cox — {ENDPOINT}: Location + Archetype (validation)",
        log_x=True,
        show=show,
    )
    return summary, forest_df


def run_lrt_partition_and_risk_grid(surv: pd.DataFrame, out_dir: Path, *, show: bool = True) -> dict[str, pd.DataFrame]:
    null_df = _cox_design(surv)
    loc_df = _cox_design(surv, ("Location",))
    arch_df = _cox_design(surv, ("Archetype",))
    full_df = _cox_design(surv, ("Location", "Archetype"))
    n_events = int(full_df[EVENT_COL].sum())

    cph_null = _fit_cox(null_df)
    cph_loc = _fit_cox(loc_df)
    cph_arch = _fit_cox(arch_df)
    cph_full = _fit_cox(full_df)

    partition_df = pd.DataFrame([_partition_loglik(
        cph_null.log_likelihood_,
        cph_loc.log_likelihood_,
        cph_arch.log_likelihood_,
        cph_full.log_likelihood_,
    )])
    partition_df.to_csv(out_dir / "cox_loglikelihood_partition.csv", index=False)

    lrt_rows = [
        _lrt_row("Null", "Location", cph_null, cph_loc, n_events),
        _lrt_row("Null", "Archetype", cph_null, cph_arch, n_events),
        _lrt_row("Null", "Location + Archetype", cph_null, cph_full, n_events),
        _lrt_row("Location", "Location + Archetype", cph_loc, cph_full, n_events),
        _lrt_row("Archetype", "Location + Archetype", cph_arch, cph_full, n_events),
    ]
    lrt_sym = pd.DataFrame(lrt_rows)
    lrt_sym.to_csv(out_dir / "cox_location_archetype_symmetric_lrt.csv", index=False)

    # S3C — symmetric LRT bar chart
    sub = (
        lrt_sym.assign(label=lambda d: d.apply(_symmetric_lrt_label, axis=1))
        .set_index("label")
        .reindex(_LRT_COMPARE_ORDER)
        .reset_index()
    )
    sub["color"] = sub.apply(_symmetric_lrt_color, axis=1)
    fig_lrt, ax_lrt = plt.subplots(figsize=(6.4, 4.0))
    y = np.arange(len(sub))
    ax_lrt.barh(y, sub["lr_statistic"], color=sub["color"], edgecolor="0.25", linewidth=0.6, height=0.62)
    xmax = max(float(sub["lr_statistic"].max()) * 1.4, 4.0)
    for yi, (_, row) in enumerate(sub.iterrows()):
        ax_lrt.text(
            row["lr_statistic"] + 0.08, yi,
            f"LR={row['lr_statistic']:.2f}, df={int(row['df'])}, {_format_lrt_p(row['p_value'])}",
            va="center", ha="left", fontsize=8,
        )
        if row["p_value"] < 0.05:
            ax_lrt.text(-0.02, yi, "*", transform=ax_lrt.get_yaxis_transform(), ha="right", va="center", fontsize=11, fontweight="bold")
    ax_lrt.set_yticks(y)
    ax_lrt.set_yticklabels(sub["label"], fontsize=8.5)
    ax_lrt.invert_yaxis()
    ax_lrt.set_xlim(0, xmax)
    ax_lrt.set_xlabel("Likelihood-ratio χ²")
    ax_lrt.set_title(f"Symmetric LRTs — {ENDPOINT} (validation, events={n_events}/{len(full_df)})", fontsize=10)
    fig_lrt.tight_layout()
    lrt_stem = out_dir / "cox_location_archetype_symmetric_lrt_OS"
    fig_lrt.savefig(lrt_stem.with_suffix(".svg"), bbox_inches="tight")
    fig_lrt.savefig(lrt_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show(fig_lrt)
    plt.close(fig_lrt)

    # S3D — prognostic information partition
    stage_order = ["chi2_unique_location", "chi2_shared", "chi2_unique_archetype"]
    stage_labels = ["Unique Location", "Shared", "Unique Archetype"]
    colors = ["#4393c3", "#878787", "#d6604d"]
    row = partition_df.iloc[0]
    fig_p, ax_p = plt.subplots(figsize=(5.5, 3.2))
    left = 0.0
    for val, lab, col in zip([row[s] for s in stage_order], stage_labels, colors):
        if val <= 0:
            continue
        ax_p.barh(0, val, left=left, height=0.45, color=col, label=lab)
        if val >= 0.8:
            ax_p.text(left + val / 2, 0, f"{val:.1f}", ha="center", va="center", fontsize=8, color="white")
        left += val
    if bool(row["shared_suppression"]):
        ax_p.text(0.5, -0.55, "shared < 0 (suppression) → clamped to 0", transform=ax_p.transAxes, fontsize=7, ha="center")
    ax_p.set_yticks([])
    ax_p.set_xlabel("Δχ² vs null model")
    ax_p.set_title(f"{ENDPOINT} (total χ² = {row['chi2_total']:.1f})")
    ax_p.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.22), fontsize=8)
    fig_p.suptitle("Prognostic information partition — Location vs Archetype (validation)", y=1.12, fontsize=11)
    fig_p.tight_layout()
    part_stem = out_dir / "cox_loglikelihood_partition"
    fig_p.savefig(part_stem.with_suffix(".svg"), bbox_inches="tight")
    fig_p.savefig(part_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show(fig_p)
    plt.close(fig_p)

    # S3C — event-rate heatmap
    from matplotlib.cm import ScalarMappable

    loc_cats = [loc for loc in LOCATION_GROUP_ORDER if loc in set(_categorical_levels(surv, "Location"))]
    arch_cats = [a for a in CLUSTER_ORDER if a in set(_categorical_levels(surv, "Archetype"))]
    risk_rows = []
    for loc in loc_cats:
        for arch in arch_cats:
            mask = (surv["Location"] == loc) & (surv["Archetype"] == arch)
            events_h, n_h, rate_h = _landmark_event_rate(surv.loc[mask])
            risk_rows.append({
                "endpoint": ENDPOINT,
                "Location": loc,
                "Archetype": arch,
                "horizon_years": RISK_GRID_HORIZON_YEARS,
                "events_2y": events_h,
                "n_at_risk_2y": n_h,
                "event_rate_2y": rate_h,
            })
    risk_grid = pd.DataFrame(risk_rows)
    risk_grid.to_csv(out_dir / "location_archetype_event_rate_grid.csv", index=False)

    mat = np.full((len(loc_cats), len(arch_cats)), np.nan)
    n_mat = np.zeros_like(mat)
    event_mat = np.zeros_like(mat)
    for i, loc in enumerate(loc_cats):
        for j, arch in enumerate(arch_cats):
            row_r = risk_grid[(risk_grid["Location"] == loc) & (risk_grid["Archetype"] == arch)]
            if len(row_r) and row_r.iloc[0]["n_at_risk_2y"] > 0:
                mat[i, j] = row_r.iloc[0]["event_rate_2y"]
                n_mat[i, j] = row_r.iloc[0]["n_at_risk_2y"]
                event_mat[i, j] = row_r.iloc[0]["events_2y"]

    vmax = float(np.nanmax(mat)) if np.isfinite(np.nanmax(mat)) and np.nanmax(mat) > 0 else 1.0
    rgba, cmap_obj, norm = _rgba_heatmap(mat, n_mat, vmin=0, vmax=vmax, use_n_alpha=False)
    fig_g, ax_g = plt.subplots(figsize=(6.5, 3.8))
    ax_g.imshow(rgba, interpolation="nearest")
    color_sm = ScalarMappable(norm=norm, cmap=cmap_obj)
    color_sm.set_array([])
    ax_g.set_xticks(np.arange(len(arch_cats)))
    ax_g.set_xticklabels([a.replace(" ", "\n") for a in arch_cats], fontsize=8)
    ax_g.set_yticks(np.arange(len(loc_cats)))
    ax_g.set_yticklabels(loc_cats, fontsize=8)
    for i in range(len(loc_cats)):
        for j in range(len(arch_cats)):
            if n_mat[i, j] > 0:
                txt = f"{int(event_mat[i, j])}/{int(n_mat[i, j])}\n({mat[i, j]:.0%})"
                ax_g.text(j, i, txt, ha="center", va="center", fontsize=7, color="white" if mat[i, j] > 0.45 else "black")
    ax_g.set_title(f"{ENDPOINT} — {RISK_GRID_HORIZON_YEARS:g}-y event rate (validation)")
    fig_g.colorbar(color_sm, ax=ax_g, fraction=0.035, pad=0.02, label=f"Event rate at {RISK_GRID_HORIZON_YEARS:g} y")
    fig_g.suptitle(
        f"Empirical outcome risk at {RISK_GRID_HORIZON_YEARS:g} y — Location × Archetype cells (validation)",
        fontsize=10,
    )
    fig_g.tight_layout()
    grid_stem = out_dir / "location_archetype_event_rate_grid"
    fig_g.savefig(grid_stem.with_suffix(".svg"), bbox_inches="tight")
    fig_g.savefig(grid_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show(fig_g)
    plt.close(fig_g)

    return {"partition": partition_df, "lrt": lrt_sym, "risk_grid": risk_grid}


def run_validation_cox_suite(
    pred: pd.DataFrame,
    out_dir: Path,
    *,
    repo_root: Path | None = None,
    write_supplementary: bool = True,
    show: bool = True,
) -> dict[str, object]:
    """Run validation Cox figures S3A–S3D (OS) and return result tables."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    surv = prepare_cox_survival(pred)
    surv.to_csv(out_dir / "cox_validation_survival_table.csv")

    results: dict[str, object] = {"surv": surv, "n_patients": len(surv), "n_events": int(surv[EVENT_COL].sum())}
    results["univariable"] = run_univariable_cox(surv, out_dir, show=show)
    results["multivariable_summary"], results["multivariable_forest"] = run_multivariable_location_archetype(
        surv, out_dir, show=show,
    )
    results.update(run_lrt_partition_and_risk_grid(surv, out_dir, show=show))

    if write_supplementary and repo_root is not None:
        from .dlbcl_io import log_saved, write_registered_supplementary_table

        log_saved(write_registered_supplementary_table(results["univariable"], repo_root, "5G_univar"), repo_root)
        log_saved(
            write_registered_supplementary_table(
                results["multivariable_forest"], repo_root, "5G_mv_forest"
            ),
            repo_root,
        )
        lrt_supp = results["lrt"].assign(label=lambda d: d.apply(_symmetric_lrt_label, axis=1))
        log_saved(write_registered_supplementary_table(lrt_supp, repo_root, "5I_sym_lrt"), repo_root)
        log_saved(write_registered_supplementary_table(results["partition"], repo_root, "5J_partition"), repo_root)
        log_saved(write_registered_supplementary_table(results["risk_grid"], repo_root, "5H_event_grid"), repo_root)

    return results


# --------------------------------------------------------------------------- #
# Within-location archetype forest plots (pooled discovery + validation)
# --------------------------------------------------------------------------- #

OS_BENCHMARK_HORIZON_YEARS = 5.0
WITHIN_LOCATION_OS_HORIZON_YEARS = OS_BENCHMARK_HORIZON_YEARS

ARCHETYPE_SHORT_LABELS = {
    "complex immune": "DI",
    "cytotoxic predominant": "CP",
    "low immune": "LO",
}

WITHIN_LOCATION_REFERENCE_GROUPS: tuple[tuple[str, str], ...] = (
    ("complex immune", "Ref DI"),
    ("cytotoxic predominant", "Ref CP"),
    ("low immune", "Ref LO"),
)


def build_pooled_discovery_validation_survival(adata, pred: pd.DataFrame, paths) -> pd.DataFrame:
    """Discovery (nb11 curative-intent) + validation OS table with harmonized labels."""
    from .km_sensitivity import build_discovery_km_survival_table, discovery_to_km_frame

    val = prepare_cox_survival(pred).copy()
    val["Cohort"] = "Validation"

    disc = discovery_to_km_frame(
        build_discovery_km_survival_table(adata, paths),
        event_col="OS_status_JV",
    ).copy()
    disc["Cohort"] = "Discovery"

    keep = [TIME_COL, EVENT_COL, "Location", "Archetype", "Cohort"]
    return pd.concat([disc[keep], val[keep]], ignore_index=True)


def _within_location_archetype_design(
    sub: pd.DataFrame,
    *,
    adjust_cohort: bool,
) -> pd.DataFrame:
    work = sub[[TIME_COL, EVENT_COL, "Archetype"] + (["Cohort"] if adjust_cohort else [])].copy()
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    if adjust_cohort and "Cohort" in work.columns and work["Cohort"].nunique() > 1:
        arch_d = pd.get_dummies(work["Archetype"], prefix="Archetype", drop_first=True)
        cohort_d = pd.get_dummies(work["Cohort"], prefix="Cohort", drop_first=True)
        return pd.concat([work[[TIME_COL, EVENT_COL]], arch_d, cohort_d], axis=1)
    return _cox_design(work, ("Archetype",))


def _archetype_dummy_term(level: str) -> str:
    return f"Archetype_{level}"


def build_within_location_archetype_forest_df(
    surv: pd.DataFrame,
    location: str,
    *,
    adjust_cohort: bool = True,
    min_patients: int = 15,
    min_events: int = 5,
) -> pd.DataFrame:
    """Forest-table rows: archetype HRs within one location, repeated for DI/CP/LO references."""
    sub = surv.loc[surv["Location"].astype(str) == location].copy()
    sub = sub.dropna(subset=[TIME_COL, EVENT_COL, "Archetype"])
    if len(sub) < min_patients or int(sub[EVENT_COL].sum()) < min_events:
        return pd.DataFrame()

    n_patients = int(len(sub))
    n_events = int(sub[EVENT_COL].sum())
    level_counts = (
        sub.groupby("Archetype", observed=True)
        .agg(n=(TIME_COL, "size"), events=(EVENT_COL, "sum"))
    )
    present_levels = [lvl for lvl in CLUSTER_ORDER if lvl in set(sub["Archetype"].astype(str))]
    rows: list[dict[str, object]] = []

    for ref_level, group_label in WITHIN_LOCATION_REFERENCE_GROUPS:
        if ref_level not in present_levels or len(present_levels) < 2:
            continue

        work = sub.copy()
        work["Archetype"] = _set_ordered_categorical(work["Archetype"], CLUSTER_ORDER, ref_level)
        try:
            cph = _fit_cox(_within_location_archetype_design(work, adjust_cohort=adjust_cohort))
        except Exception:
            continue

        summary = cph.summary
        ref_short = ARCHETYPE_SHORT_LABELS[ref_level]
        for level in present_levels:
            short = ARCHETYPE_SHORT_LABELS[level]
            n_level = int(level_counts.loc[level, "n"]) if level in level_counts.index else np.nan
            evt_level = int(level_counts.loc[level, "events"]) if level in level_counts.index else np.nan
            label = f"{short} vs {ref_short} (n={n_level}, {evt_level} evt)"
            if level == ref_level:
                rows.append(
                    {
                        "factor": group_label,
                        "label": f"{short} (reference)",
                        "n_patients": n_patients,
                        "n_events": n_events,
                        "HR": 1.0,
                        "CI_lower": 1.0,
                        "CI_upper": 1.0,
                        "p": np.nan,
                        "is_reference": True,
                    }
                )
                continue

            term = _archetype_dummy_term(level)
            if term not in summary.index:
                continue
            row = summary.loc[term]
            rows.append(
                {
                    "factor": group_label,
                    "label": label,
                    "n_patients": n_patients,
                    "n_events": n_events,
                    "HR": float(np.exp(row["coef"])),
                    "CI_lower": float(np.exp(row["coef lower 95%"])),
                    "CI_upper": float(np.exp(row["coef upper 95%"])),
                    "p": float(row["p"]),
                    "is_reference": False,
                }
            )

    return pd.DataFrame(rows)


DI_REFERENCE_ARCHETYPE = "complex immune"
VS_DI_CONTRAST_LEVELS: tuple[tuple[str, str], ...] = (
    ("cytotoxic predominant", "CP vs DI"),
    ("low immune", "LO vs DI"),
)


def build_location_archetype_vs_di_forest_df(
    surv: pd.DataFrame,
    *,
    adjust_cohort: bool = True,
    horizon_years: float = WITHIN_LOCATION_OS_HORIZON_YEARS,
    min_patients: int = 15,
    min_events: int = 5,
) -> pd.DataFrame:
    """One row per location × contrast (CP vs DI, LO vs DI); 5-year landmark OS by default."""
    from .cox_forest_plot import add_fdr_column_by_group

    surv = apply_os_landmark_censoring(surv, horizon_years=horizon_years)
    rows: list[dict[str, object]] = []
    for loc in LOCATION_GROUP_ORDER:
        if loc not in set(surv["Location"].dropna().astype(str)):
            continue

        sub = surv.loc[surv["Location"].astype(str) == loc].copy()
        sub = sub.dropna(subset=[TIME_COL, EVENT_COL, "Archetype"])
        if len(sub) < min_patients or int(sub[EVENT_COL].sum()) < min_events:
            continue

        present_levels = [lvl for lvl in CLUSTER_ORDER if lvl in set(sub["Archetype"].astype(str))]
        if DI_REFERENCE_ARCHETYPE not in present_levels or len(present_levels) < 2:
            continue

        level_counts = (
            sub.groupby("Archetype", observed=True)
            .agg(n=(TIME_COL, "size"), events=(EVENT_COL, "sum"))
        )

        work = sub.copy()
        work["Archetype"] = _set_ordered_categorical(work["Archetype"], CLUSTER_ORDER, DI_REFERENCE_ARCHETYPE)
        try:
            cph = _fit_cox(_within_location_archetype_design(work, adjust_cohort=adjust_cohort))
        except Exception:
            continue

        summary = cph.summary
        for level, contrast in VS_DI_CONTRAST_LEVELS:
            if level not in present_levels:
                continue
            term = _archetype_dummy_term(level)
            if term not in summary.index:
                continue
            est = summary.loc[term]
            rows.append(
                {
                    "Location": loc,
                    "contrast": contrast,
                    "archetype": level,
                    "reference": DI_REFERENCE_ARCHETYPE,
                    "horizon_years": float(horizon_years),
                    "n_location": int(len(sub)),
                    "n_events_location": int(sub[EVENT_COL].sum()),
                    "n_archetype": int(level_counts.loc[level, "n"]),
                    "n_events_archetype": int(level_counts.loc[level, "events"]),
                    "HR": float(np.exp(est["coef"])),
                    "CI_lower": float(np.exp(est["coef lower 95%"])),
                    "CI_upper": float(np.exp(est["coef upper 95%"])),
                    "p": float(est["p"]),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return add_fdr_column_by_group(df, "Location", p_col="p")


def save_location_archetype_vs_di_forest_plot(
    surv: pd.DataFrame,
    out_stem: Path | str,
    *,
    adjust_cohort: bool = True,
    horizon_years: float = WITHIN_LOCATION_OS_HORIZON_YEARS,
    show: bool = False,
) -> pd.DataFrame:
    """Site-stratified forest: anatomical location rows, CP/LO vs DI (5-year landmark OS)."""
    from .cox_forest_plot import save_location_archetype_vs_di_forest

    plot_df = build_location_archetype_vs_di_forest_df(
        surv,
        adjust_cohort=adjust_cohort,
        horizon_years=horizon_years,
    )
    if plot_df.empty:
        raise ValueError("no location × archetype contrasts available for DI-referenced forest")

    cohort_note = "cohort-adjusted" if adjust_cohort else "unadjusted"
    title = (
        f"{ENDPOINT} at {horizon_years:g} y — archetype hazard vs DI (complex immune) by anatomical site "
        f"(Discovery+Validation, {cohort_note})"
    )
    return save_location_archetype_vs_di_forest(
        plot_df,
        Path(out_stem),
        location_order=LOCATION_GROUP_ORDER,
        title=title,
        show=show,
    )


def save_within_location_archetype_forest_plots(
    surv: pd.DataFrame,
    out_dir: Path | str,
    *,
    locations: tuple[str, ...] | None = None,
    adjust_cohort: bool = True,
    show: bool = False,
) -> dict[str, pd.DataFrame]:
    """Write one forest plot per location (DI/CP/LO reference groups)."""
    from .cox_forest_plot import save_cox_forest_table

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if locations is None:
        locations = tuple(x for x in LOCATION_GROUP_ORDER if x in set(surv["Location"].dropna().astype(str)))

    results: dict[str, pd.DataFrame] = {}
    for loc in locations:
        plot_df = build_within_location_archetype_forest_df(surv, loc, adjust_cohort=adjust_cohort)
        if plot_df.empty:
            continue

        sub = surv.loc[surv["Location"].astype(str) == loc]
        n_pat, n_evt = int(len(sub)), int(sub[EVENT_COL].sum())
        cohort_note = "cohort-adjusted" if adjust_cohort else "unadjusted"
        title = (
            f"{ENDPOINT} within {loc} — archetype HR by reference "
            f"(Discovery+Validation, {cohort_note}; n={n_pat}, events={n_evt})"
        )
        stem = out_dir / f"within_{loc.lower()}_archetype_reference_forest_OS"
        plot_df.to_csv(stem.with_name(f"{stem.name}_table.csv"), index=False)
        save_cox_forest_table(
            plot_df,
            stem,
            title=title,
            log_x=True,
            show_fdr=False,
            show=show,
        )
        results[loc] = plot_df
    return results


# --------------------------------------------------------------------------- #
# Archetype OS benchmark table (validation + published cohorts)
# --------------------------------------------------------------------------- #

DIVERSE_IMMUNE_ARCHETYPE = "complex immune"
LO_CP_ARCHETYPES = ("low immune", "cytotoxic predominant")

# 5-year OS univariable Cox: diverse immune (complex immune) vs LO+CP reference.
ARCHETYPE_OS_BENCHMARK_ROWS: tuple[dict[str, object], ...] = (
    {
        "Dataset": "Validation",
        "n": 308,
        "Events": 98,
        "HR": 2.15,
        "ci_lower": 1.19,
        "ci_upper": 3.90,
        "p": 0.0116,
    },
    {
        "Dataset": "Schmitz",
        "n": 234,
        "Events": 79,
        "HR": 2.08,
        "ci_lower": 1.21,
        "ci_upper": 3.55,
        "p": 0.00764,
    },
    {
        "Dataset": "Chapuy",
        "n": 103,
        "Events": 28,
        "HR": 2.93,
        "ci_lower": 1.02,
        "ci_upper": 8.44,
        "p": 0.0467,
    },
    {
        "Dataset": "Ennishi",
        "n": 296,
        "Events": 95,
        "HR": 2.89,
        "ci_lower": 1.67,
        "ci_upper": 5.02,
        "p": 1.61e-4,
    },
)


def apply_os_landmark_censoring(
    surv: pd.DataFrame,
    *,
    horizon_years: float = OS_BENCHMARK_HORIZON_YEARS,
) -> pd.DataFrame:
    """Landmark OS at ``horizon_years`` (censor beyond horizon)."""
    out = surv.copy()
    time = pd.to_numeric(out[TIME_COL], errors="coerce")
    event = pd.to_numeric(out[EVENT_COL], errors="coerce")
    out.loc[time > horizon_years, EVENT_COL] = 0
    out.loc[time > horizon_years, TIME_COL] = horizon_years
    return out.loc[time > 0].copy()


def compute_validation_diverse_immune_benchmark_row(
    pred: pd.DataFrame,
    *,
    horizon_years: float = OS_BENCHMARK_HORIZON_YEARS,
) -> dict[str, object]:
    """5-year univariable Cox: diverse immune (complex immune) vs LO+CP (QC helper)."""
    surv = apply_os_landmark_censoring(prepare_cox_survival(pred), horizon_years=horizon_years)
    diverse = surv["Archetype"].eq(DIVERSE_IMMUNE_ARCHETYPE).astype(int)
    df = pd.DataFrame(
        {
            TIME_COL: surv[TIME_COL],
            EVENT_COL: surv[EVENT_COL],
            "diverse_immune": diverse,
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    cph = CoxPHFitter(penalizer=PENALIZER)
    cph.fit(df, duration_col=TIME_COL, event_col=EVENT_COL)
    row = cph.summary.loc["diverse_immune"]
    return {
        "Dataset": "Validation (computed)",
        "n": int(len(df)),
        "HR": float(np.exp(row["coef"])),
        "p": float(row["p"]),
    }


def _format_benchmark_cell(value: object, *, kind: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if kind == "n":
        return str(int(value))
    if kind == "hr":
        return f"{float(value):.2f}"
    if kind == "p":
        p = float(value)
        if p < 0.001:
            return f"{p:.2e}"
        return f"{p:g}"
    raise ValueError(f"unknown benchmark cell kind: {kind}")


def build_archetype_os_benchmark_table(
    rows: tuple[dict[str, object], ...] | None = None,
) -> pd.DataFrame:
    """Assemble the archetype OS benchmark table (manuscript values by default)."""
    data = rows or ARCHETYPE_OS_BENCHMARK_ROWS
    return pd.DataFrame(data, columns=["Dataset", "n", "HR", "p"])


def build_archetype_os_benchmark_forest_df(
    rows: tuple[dict[str, object], ...] | None = None,
    *,
    include_pooled: bool = True,
    pooled_label: str = "Overall",
) -> pd.DataFrame:
    """Forest-plot dataframe with HR confidence intervals and event counts."""
    from .cox_forest_plot import append_pooled_cohort_row

    data = rows or ARCHETYPE_OS_BENCHMARK_ROWS
    df = pd.DataFrame(data).rename(columns={"ci_lower": "CI_lower", "ci_upper": "CI_upper"})
    if include_pooled:
        df = append_pooled_cohort_row(df, label=pooled_label)
    return df


def render_archetype_os_benchmark_great_table(
    table_df: pd.DataFrame,
    *,
    title: str = "Immune archetype prognostic association (OS)",
    subtitle: str | None = None,
) -> object:
    from great_tables import GT, loc, style

    if subtitle is None:
        subtitle = (
            f"{int(OS_BENCHMARK_HORIZON_YEARS)}-year univariable Cox; "
            "diverse immune (complex immune) vs LO+CP (low immune + cytotoxic predominant)"
        )

    display_df = pd.DataFrame(
        {
            "Dataset": table_df["Dataset"].astype(str),
            "n": [_format_benchmark_cell(v, kind="n") for v in table_df["n"]],
            "HR": [_format_benchmark_cell(v, kind="hr") for v in table_df["HR"]],
            "p": [_format_benchmark_cell(v, kind="p") for v in table_df["p"]],
        }
    )

    return (
        GT(display_df)
        .tab_header(title=title, subtitle=subtitle)
        .cols_label(Dataset="Dataset", n="n", HR="HR", p="p")
        .tab_style(style=style.text(weight="bold"), locations=loc.column_labels())
        .cols_align(align="left", columns=["Dataset"])
        .cols_align(align="center", columns=["n", "HR", "p"])
        .tab_options(
            table_width="100%",
            table_font_size="11px",
            heading_title_font_size="14px",
        )
    )


def save_archetype_os_benchmark_table(
    table_df: pd.DataFrame,
    output_dir: Path | str,
    *,
    stem: str = "archetype_os_cohort_benchmark",
    title: str = "Immune archetype prognostic association (OS)",
    subtitle: str | None = None,
) -> dict[str, Path | object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path | object] = {
        "csv": output_dir / f"{stem}.csv",
        "html": output_dir / f"{stem}.html",
    }
    table_df.to_csv(paths["csv"], index=False)
    gt_table = render_archetype_os_benchmark_great_table(table_df, title=title, subtitle=subtitle)
    paths["html"].write_text(gt_table.as_raw_html(make_page=True), encoding="utf-8")  # type: ignore[union-attr]
    png_path = output_dir / f"{stem}.png"
    try:
        gt_table.gtsave(png_path)
        paths["png"] = png_path
    except Exception:
        pass
    paths["gt"] = gt_table
    return paths


def save_archetype_os_benchmark_forest(
    forest_df: pd.DataFrame,
    output_dir: Path | str,
    *,
    stem: str = "archetype_os_cohort_forest",
    title: str = "Immune archetype prognostic association (OS)",
    subtitle: str | None = None,
    show: bool = False,
) -> dict[str, Path]:
    from .cox_forest_plot import save_cohort_hr_forest

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if subtitle is None:
        subtitle = (
            f"{int(OS_BENCHMARK_HORIZON_YEARS)}-year univariable Cox; "
            "diverse immune (complex immune) vs LO+CP (low immune + cytotoxic predominant)"
        )
    full_title = f"{title}\n{subtitle}" if subtitle else title
    out_stem = output_dir / stem
    save_cohort_hr_forest(
        forest_df,
        out_stem,
        title=full_title,
        log_x=True,
        include_pooled=False,
        show=show,
    )
    return {
        "svg": out_stem.with_suffix(".svg"),
        "png": out_stem.with_suffix(".png"),
        "csv": out_stem.parent / f"{stem}_table.csv",
    }
