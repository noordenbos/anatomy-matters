"""IPI / MSKCC scoring and risk bucketing for discovery and validation cohorts.

Systemic DLBCL uses IPI (5 factors). PCNSL uses the MSKCC class score (1–3)
when available (replacing IELSG for primary risk bucketing). Patients with
incomplete IPI factors are still assigned when the known bounds force a bucket.

Primary bucket:   low ``0-2`` vs high ``>=3``
  (MSKCC class 1–2 → ``0-2``; class 3 → ``>=3``)
Secondary bucket: low ``0-1`` vs high ``>=2``
Tertiary bucket:  low ``0-3`` vs high ``>3``
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

N_IPI_FACTORS = 5
N_IELSG_FACTORS = 5

BUCKET_PRIMARY_LOW = "0-2"
BUCKET_PRIMARY_HIGH = ">=3"
BUCKET_SECONDARY_LOW = "0-1"
BUCKET_SECONDARY_HIGH = ">=2"
BUCKET_TERTIARY_LOW = "0-3"
BUCKET_TERTIARY_HIGH = ">3"

PRIMARY_CATEGORIES = (BUCKET_PRIMARY_LOW, BUCKET_PRIMARY_HIGH)
SECONDARY_CATEGORIES = (BUCKET_SECONDARY_LOW, BUCKET_SECONDARY_HIGH)
TERTIARY_CATEGORIES = (BUCKET_TERTIARY_LOW, BUCKET_TERTIARY_HIGH)

_PCNSL_MARKERS = frozenset({"cns", "pcns", "pcnsl", "brain"})
_MISSING_TEXT = frozenset({"", "nan", "none", "na", "unk", "unknown", "<na>", "nat"})


def _is_missing(value: object) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return True
    if isinstance(value, (pd.Timestamp, np.datetime64)) and pd.isna(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in _MISSING_TEXT


def _is_pcnsl_row(
    *,
    disease_type: object = None,
    location: object = None,
    ann_arbor: object = None,
    origin: object = None,
) -> bool:
    for value in (disease_type, location, origin, ann_arbor):
        if _is_missing(value):
            continue
        text = str(value).strip().lower()
        if text in _PCNSL_MARKERS or text.startswith("pcnsl"):
            return True
    return False


def _binary_yes_no_unknown(value: object) -> int | None:
    """Map 0/'no'→0, 1/'yes'→1, 2/'unknown'/missing→None."""
    if _is_missing(value):
        return None
    text = str(value).strip().lower()
    if text in {"0", "0.0", "no", "n", "false", "neg", "negative", "normal"}:
        return 0
    if text in {"1", "1.0", "yes", "y", "true", "pos", "positive", "elevated", "high"}:
        return 1
    if text in {"2", "2.0", "unknown", "unk", "u"}:
        return None
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return None
    if float(num) == 0.0:
        return 0
    if float(num) == 1.0:
        return 1
    if float(num) == 2.0:
        return None
    return None


def age_risk_point(age: object, *, threshold: float = 60.0) -> int | None:
    if _is_missing(age):
        return None
    num = pd.to_numeric(age, errors="coerce")
    if pd.isna(num):
        return None
    return 1 if float(num) > threshold else 0


def ann_arbor_risk_point(ann_arbor: object) -> int | None:
    """IPI stage factor: +1 for Ann Arbor III or IV."""
    if _is_missing(ann_arbor):
        return None
    text = str(ann_arbor).strip().upper()
    if text in {"PCNSL"} or text.startswith("PCNSL"):
        return None
    # Roman / arabic stage token at start (III before II before I).
    if text.startswith("IV") or text.startswith("4"):
        return 1
    if text.startswith("III") or text.startswith("3"):
        return 1
    if text.startswith("II") or text.startswith("2"):
        return 0
    if text.startswith("I") or text.startswith("1"):
        return 0
    return None


def extranodal_risk_point(value: object) -> int | None:
    """IPI extranodal factor: +1 if >1 extranodal site.

    Workbook ``ipi_extranodal`` is typically already the binary factor (0/1).
    Numeric counts >1 are also treated as adverse.
    """
    if _is_missing(value):
        return None
    text = str(value).strip().lower()
    if text in {"2", "2.0", "unknown", "unk", "u"}:
        return None
    if text in {"0", "0.0", "no", "n", "false"}:
        return 0
    if text in {"1", "1.0", "yes", "y", "true"}:
        return 1
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return None
    if float(num) > 1.0:
        return 1
    if float(num) == 1.0:
        return 1
    if float(num) == 0.0:
        return 0
    return None


def who_ecog_risk_point(value: object) -> int | None:
    """IPI performance-status factor: +1 if ECOG/WHO >= 2."""
    if _is_missing(value):
        return None
    text = str(value).strip().lower()
    if text in {"unknown", "unk", "u"}:
        return None
    if text in {"yes", "y", "true"}:
        return 1
    if text in {"no", "n", "false"}:
        return 0
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return None
    # ECOG/WHO 0–5: adverse if >=2 (2 is a valid score, not an unknown code).
    if 0.0 <= float(num) <= 5.0:
        return 1 if float(num) >= 2.0 else 0
    return None


def ldh_risk_point(value: object) -> int | None:
    """IPI LDH factor (workbook ``ipi_ldh``): 0/1 known, 2=unknown."""
    return _binary_yes_no_unknown(value)


@dataclass(frozen=True)
class FactorScore:
    """Known adverse-factor count with incomplete-information bounds."""

    points: tuple[int | None, ...]
    system: str  # "IPI" | "MSKCC" | "IELSG"
    exact_total: float | None = None  # recorded full score when available

    @property
    def n_factors(self) -> int:
        return len(self.points)

    @property
    def n_known(self) -> int:
        return sum(p is not None for p in self.points)

    @property
    def n_unknown(self) -> int:
        return sum(p is None for p in self.points)

    @property
    def n_positive(self) -> int:
        return sum(1 for p in self.points if p == 1)

    @property
    def n_negative(self) -> int:
        return sum(1 for p in self.points if p == 0)

    @property
    def min_score(self) -> int:
        if self.exact_total is not None and np.isfinite(self.exact_total):
            return int(self.exact_total)
        return self.n_positive

    @property
    def max_score(self) -> int:
        if self.exact_total is not None and np.isfinite(self.exact_total):
            return int(self.exact_total)
        return self.n_positive + self.n_unknown

    @property
    def exact_score(self) -> int | None:
        if self.exact_total is not None and np.isfinite(self.exact_total):
            return int(self.exact_total)
        if self.n_unknown == 0:
            return self.n_positive
        return None


def bucket_from_bounds(
    min_score: int,
    max_score: int,
    *,
    low_max: int,
    high_min: int,
) -> str | None:
    """Assign low/high when bounds cannot cross the cut; else None."""
    if max_score <= low_max:
        return f"0-{low_max}" if low_max >= 1 else "0"
    if min_score >= high_min:
        return f">={high_min}"
    return None


def primary_bucket(min_score: int, max_score: int) -> str | None:
    return bucket_from_bounds(min_score, max_score, low_max=2, high_min=3)


def secondary_bucket(min_score: int, max_score: int) -> str | None:
    return bucket_from_bounds(min_score, max_score, low_max=1, high_min=2)


def tertiary_bucket(min_score: int, max_score: int) -> str | None:
    """Low ``0-3`` vs high ``>3`` (i.e. high when min score >= 4)."""
    out = bucket_from_bounds(min_score, max_score, low_max=3, high_min=4)
    if out == ">=4":
        return BUCKET_TERTIARY_HIGH
    return out


def score_ipi_factors(row: pd.Series) -> FactorScore:
    points = (
        age_risk_point(row.get("age")),
        ann_arbor_risk_point(row.get("Ann_Arbor_at_Dx", row.get("ann_arbor"))),
        ldh_risk_point(row.get("ipi_ldh")),
        who_ecog_risk_point(row.get("ipi_who")),
        extranodal_risk_point(row.get("ipi_extranodal")),
    )
    exact = pd.to_numeric(row.get("ipi_score"), errors="coerce")
    exact_total = float(exact) if pd.notna(exact) else None
    return FactorScore(points=points, system="IPI", exact_total=exact_total)


def score_ielsg_factors(row: pd.Series) -> FactorScore:
    """Legacy IELSG component scoring (kept for audits; not used for primary buckets)."""
    points = (
        age_risk_point(row.get("age")),
        _binary_yes_no_unknown(row.get("ielsg_ldh")),
        _binary_yes_no_unknown(row.get("ielsg_deepbrain")),
        _binary_yes_no_unknown(row.get("ielsg_ecog")),
        _binary_yes_no_unknown(row.get("ielsg_csf")),
    )
    return FactorScore(points=points, system="IELSG", exact_total=None)


def mskcc_class(value: object) -> int | None:
    """Parse MSKCC PCNSL class ``1``/``2``/``3``; unknown/blank → None."""
    if _is_missing(value):
        return None
    text = str(value).strip().lower()
    if text in {"unknown", "unk", "u"}:
        return None
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return None
    clas = int(num)
    if clas in {1, 2, 3}:
        return clas
    return None


def score_mskcc_factors(row: pd.Series) -> FactorScore:
    """PCNSL MSKCC class as an exact score (1–3) for primary/secondary bucketing."""
    clas = mskcc_class(row.get("MSKCC", row.get("mskcc")))
    if clas is None:
        # Three unknowns → bounds 0–3 cross the primary cut → unassignable.
        return FactorScore(points=(None, None, None), system="MSKCC", exact_total=None)
    return FactorScore(points=(clas,), system="MSKCC", exact_total=float(clas))


def score_patient(row: pd.Series) -> FactorScore:
    if _is_pcnsl_row(
        disease_type=row.get("disease_type"),
        location=row.get("Location", row.get("location")),
        ann_arbor=row.get("Ann_Arbor_at_Dx"),
        origin=row.get("origin"),
    ):
        return score_mskcc_factors(row)
    return score_ipi_factors(row)


def assign_ipi_ielsg_buckets(pred: pd.DataFrame) -> pd.DataFrame:
    """Compute IPI/MSKCC factor bounds and primary/secondary risk buckets.

    Returns a frame aligned to ``pred.index`` with:
    ``system``, ``score_exact``, ``score_min``, ``score_max``,
    ``n_known_factors``, ``n_unknown_factors``,
    ``bucket_primary``, ``bucket_secondary``.
    """
    rows: list[dict[str, object]] = []
    for idx, row in pred.iterrows():
        scored = score_patient(row)
        rows.append(
            {
                "patient_id": idx,
                "system": scored.system,
                "score_exact": scored.exact_score if scored.exact_score is not None else pd.NA,
                "score_min": scored.min_score,
                "score_max": scored.max_score,
                "n_known_factors": scored.n_known,
                "n_unknown_factors": scored.n_unknown,
                "n_positive_factors": scored.n_positive,
                "n_negative_factors": scored.n_negative,
                "bucket_primary": primary_bucket(scored.min_score, scored.max_score),
                "bucket_secondary": secondary_bucket(scored.min_score, scored.max_score),
                "bucket_tertiary": tertiary_bucket(scored.min_score, scored.max_score),
            }
        )
    out = pd.DataFrame(rows).set_index("patient_id")
    out.index = out.index.astype(str)
    out.index.name = pred.index.name
    return out


def attach_ipi_ielsg_to_survival(
    surv: pd.DataFrame,
    pred: pd.DataFrame,
    *,
    primary_col: str = "IPI_score",
    secondary_col: str = "IPI_score_secondary",
    tertiary_col: str = "IPI_score_tertiary",
    include_secondary: bool = True,
    include_tertiary: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Attach primary (and optional secondary/tertiary) IPI/IELSG buckets to a Cox table.

    Primary is the default baseline covariate (``0-2`` vs ``>=3``).
    """
    out = surv.copy()
    pred = pred.copy()
    pred.index = pred.index.astype(str)
    out.index = out.index.astype(str)

    # Bring Location / disease_type into scoring row when present only on surv.
    score_input = pred.reindex(out.index).copy()
    if "Location" in out.columns and "Location" not in score_input.columns:
        score_input["Location"] = out["Location"]
    if "disease_type" not in score_input.columns and "Location" in out.columns:
        # PCNS detection also uses Location on surv.
        pass

    buckets = assign_ipi_ielsg_buckets(score_input)
    buckets = buckets.reindex(out.index)

    extra: list[str] = []
    primary = buckets["bucket_primary"]
    if primary.notna().any():
        out[primary_col] = pd.Categorical(
            primary,
            categories=list(PRIMARY_CATEGORIES),
            ordered=False,
        )
        extra.append(primary_col)

    if include_secondary:
        secondary = buckets["bucket_secondary"]
        if secondary.notna().any():
            out[secondary_col] = pd.Categorical(
                secondary,
                categories=list(SECONDARY_CATEGORIES),
                ordered=False,
            )

    if include_tertiary:
        tertiary = buckets["bucket_tertiary"]
        if tertiary.notna().any():
            out[tertiary_col] = pd.Categorical(
                tertiary,
                categories=list(TERTIARY_CATEGORIES),
                ordered=False,
            )

    for col in (
        "system",
        "score_exact",
        "score_min",
        "score_max",
        "n_known_factors",
        "n_unknown_factors",
    ):
        out[f"ipi_ielsg_{col}"] = buckets[col]

    return out, extra


def _fmt_count_pct(count: int, denom: int) -> str:
    if denom <= 0:
        return f"{count} (n/a)"
    return f"{count} ({100.0 * count / denom:.1f}%)"


def coverage_exact_vs_bucket(pred: pd.DataFrame) -> pd.DataFrame:
    """Compare exact-integer vs partial-bucket coverage for each cut.

    Display columns: ``System``, ``Exact integer``, ``in bucket``, ``n low``,
    ``n high`` — counts with % of system ``n``. Numeric columns are also kept
    for downstream use (``*_n`` / ``n``).
    """
    buckets = assign_ipi_ielsg_buckets(pred)
    schemes = (
        ("secondary (0-1 vs >=2)", "bucket_secondary", BUCKET_SECONDARY_LOW, BUCKET_SECONDARY_HIGH),
        ("primary (0-2 vs >=3)", "bucket_primary", BUCKET_PRIMARY_LOW, BUCKET_PRIMARY_HIGH),
        ("tertiary (0-3 vs >3)", "bucket_tertiary", BUCKET_TERTIARY_LOW, BUCKET_TERTIARY_HIGH),
    )
    rows: list[dict[str, object]] = []
    systems = ["IPI", "MSKCC", "ALL"]
    for scheme_label, col, low_lab, high_lab in schemes:
        for system in systems:
            sub = buckets if system == "ALL" else buckets.loc[buckets["system"] == system]
            n = len(sub)
            n_exact = int(sub["score_exact"].notna().sum())
            n_in = int(sub[col].notna().sum())
            n_low = int((sub[col] == low_lab).sum())
            n_high = int((sub[col] == high_lab).sum())
            rows.append(
                {
                    "Scheme": scheme_label,
                    "System": system,
                    "Exact integer": _fmt_count_pct(n_exact, n),
                    "in bucket": _fmt_count_pct(n_in, n),
                    "n low": _fmt_count_pct(n_low, n),
                    "n high": _fmt_count_pct(n_high, n),
                    "n": n,
                    "Exact integer_n": n_exact,
                    "in bucket_n": n_in,
                    "n low_n": n_low,
                    "n high_n": n_high,
                }
            )
    return pd.DataFrame(rows)


def summarize_ipi_ielsg_buckets(pred: pd.DataFrame) -> pd.DataFrame:
    """Coverage table for QC (counts by system × primary/secondary/tertiary)."""
    buckets = assign_ipi_ielsg_buckets(pred)
    rows = []
    for system, sub in buckets.groupby("system"):
        rows.append(
            {
                "system": system,
                "n": len(sub),
                "primary_assigned": int(sub["bucket_primary"].notna().sum()),
                "primary_low": int((sub["bucket_primary"] == BUCKET_PRIMARY_LOW).sum()),
                "primary_high": int((sub["bucket_primary"] == BUCKET_PRIMARY_HIGH).sum()),
                "secondary_assigned": int(sub["bucket_secondary"].notna().sum()),
                "secondary_low": int((sub["bucket_secondary"] == BUCKET_SECONDARY_LOW).sum()),
                "secondary_high": int((sub["bucket_secondary"] == BUCKET_SECONDARY_HIGH).sum()),
                "tertiary_assigned": int(sub["bucket_tertiary"].notna().sum()),
                "tertiary_low": int((sub["bucket_tertiary"] == BUCKET_TERTIARY_LOW).sum()),
                "tertiary_high": int((sub["bucket_tertiary"] == BUCKET_TERTIARY_HIGH).sum()),
                "exact_score_known": int(sub["score_exact"].notna().sum()),
                "mean_unknown_factors": float(sub["n_unknown_factors"].mean()),
            }
        )
    total = {
        "system": "ALL",
        "n": len(buckets),
        "primary_assigned": int(buckets["bucket_primary"].notna().sum()),
        "primary_low": int((buckets["bucket_primary"] == BUCKET_PRIMARY_LOW).sum()),
        "primary_high": int((buckets["bucket_primary"] == BUCKET_PRIMARY_HIGH).sum()),
        "secondary_assigned": int(buckets["bucket_secondary"].notna().sum()),
        "secondary_low": int((buckets["bucket_secondary"] == BUCKET_SECONDARY_LOW).sum()),
        "secondary_high": int((buckets["bucket_secondary"] == BUCKET_SECONDARY_HIGH).sum()),
        "tertiary_assigned": int(buckets["bucket_tertiary"].notna().sum()),
        "tertiary_low": int((buckets["bucket_tertiary"] == BUCKET_TERTIARY_LOW).sum()),
        "tertiary_high": int((buckets["bucket_tertiary"] == BUCKET_TERTIARY_HIGH).sum()),
        "exact_score_known": int(buckets["score_exact"].notna().sum()),
        "mean_unknown_factors": float(buckets["n_unknown_factors"].mean()),
    }
    return pd.DataFrame([*rows, total])
