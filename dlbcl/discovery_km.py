"""Discovery-cohort Kaplan–Meier survival (nb11 logic, shared by compiled fig1/fig2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test, multivariate_logrank_test

from .dlbcl_io import log_wrote

ENDPOINTS = {
    "OS": {"time": "OS_time_JV", "event": "OS_status_JV"},
    "DSS": {"time": "OS_time_JV", "event": "OS_specific_status_JV"},
    "PFS": {"time": "pfs_time", "event": "pfs_status"},
}
REQUIRED_ENDPOINTS = ("OS", "DSS")
KM_HORIZON_YEARS = 5.0
LOCATION_KM_FIGSIZE = (12, 5)
ARCHETYPE_KM_FIGSIZE = (9, 6)
KM_AT_RISK_BOTTOM = 0.28

ARCHETYPE_NAME_MAP = {
    1: "low immune",
    2: "cytotoxic predominant",
    3: "complex immune",
}
LOCATION_RECODE = {
    "pBONE": "bone",
    "polyOST": "bone",
    "disseminated": "bone",
}
LOCATION_ORDER = ["PCNS", "nodal", "bone", "testis"]
LOCATION_COLORS = {
    "PCNS": "#1a2252",
    "nodal": "#2fa148",
    "bone": "#f57f20",
    "testis": "#d62a28",
}
ARCHETYPE_ORDER = [1, 2, 3]
ARCHETYPE_COLORS = {
    1: "#d73027",
    2: "#4575b4",
    3: "#1a9850",
}

_REPO_ROOT: Path | None = None


def configure_km_runtime(repo_root: Path) -> None:
    """Set repo root for ``log_wrote`` inside KM plot functions."""
    global _REPO_ROOT  # noqa: PLW0603
    _REPO_ROOT = repo_root


def _patient_frame(adata: Any, uns_key: str) -> pd.DataFrame:
    df = pd.DataFrame(adata.uns[uns_key]).copy()
    if "patient_id" not in df.columns:
        df = df.reset_index(names="patient_id")
    df["patient_id"] = df["patient_id"].astype(str)
    return df.set_index("patient_id")


def build_discovery_survival_table(
    adata: Any,
    arch_df: pd.DataFrame,
    *,
    km_dir: Path | None = None,
) -> pd.DataFrame:
    """Curative-intent discovery KM cohort from ``adata.uns`` + archetype assignments."""
    clinical = _patient_frame(adata, "case_clinical")
    classif = _patient_frame(adata, "case_classifications")

    clinical["Curative_intent"] = pd.to_numeric(clinical["Curative_intent"], errors="coerce")
    time_cols = list(dict.fromkeys(ep["time"] for ep in ENDPOINTS.values()))
    event_cols = [ep["event"] for ep in ENDPOINTS.values()]
    surv = clinical.loc[
        clinical["Curative_intent"] == 1,
        list(dict.fromkeys([*time_cols, *event_cols])),
    ].copy()

    surv["Location"] = classif["Location"].replace(LOCATION_RECODE)

    cluster_dict = dict(
        zip(arch_df["patient_id"].astype(str), arch_df["abundance_cluster_30"].astype(int))
    )
    arch = pd.Series(cluster_dict, name="archetype_id")
    surv["archetype_id"] = arch
    surv["Archetype"] = surv["archetype_id"].map(ARCHETYPE_NAME_MAP)

    required_cols = [
        ENDPOINTS[ep]["time"] for ep in REQUIRED_ENDPOINTS
    ] + [ENDPOINTS[ep]["event"] for ep in REQUIRED_ENDPOINTS]
    numeric_cols = list(dict.fromkeys([*time_cols, *event_cols, "archetype_id"]))
    for col in numeric_cols:
        surv[col] = pd.to_numeric(surv[col], errors="coerce")

    surv = surv.dropna(subset=[*required_cols, "Location", "Archetype"])

    if km_dir is not None:
        km_dir.mkdir(parents=True, exist_ok=True)
        surv.to_csv(km_dir / "km_survival_table.csv")

    return surv


def _format_pvalue(p: float) -> str:
    if np.isnan(p):
        return "NA"
    if p < 0.0001:
        return "<0.0001"
    return f"{p:.4f}"


def _prepare_km_frame(
    df: pd.DataFrame,
    *,
    time_col: str,
    event_col: str,
    horizon: float,
) -> pd.DataFrame:
    out = df.copy()
    out["time"] = pd.to_numeric(out[time_col], errors="coerce")
    out["event"] = pd.to_numeric(out[event_col], errors="coerce") == 1
    out = out.dropna(subset=["time"])
    out = out[out["time"] > 0].copy()
    out.loc[out["time"] > horizon, "event"] = False
    out.loc[out["time"] > horizon, "time"] = horizon
    return out


def _add_censor_ticks(ax, group_data: pd.DataFrame, color: str, kmf: KaplanMeierFitter) -> None:
    censored = group_data[~group_data["event"]]
    for _, row in censored.iterrows():
        survival_prob = float(kmf.predict(row["time"]))
        ax.plot(
            row["time"],
            survival_prob,
            marker="|",
            color=color,
            markersize=12,
            markeredgewidth=2.5,
            linestyle="None",
            zorder=5,
        )


def plot_km_by_location(
    df: pd.DataFrame,
    *,
    time_col: str,
    event_col: str,
    title: str,
    ylabel: str,
    out_path: Path,
    horizon: float = KM_HORIZON_YEARS,
) -> pd.DataFrame:
    plot_df = _prepare_km_frame(df, time_col=time_col, event_col=event_col, horizon=horizon)
    plot_df = plot_df[plot_df["Location"].isin(LOCATION_ORDER)].copy()

    fig, ax = plt.subplots(figsize=LOCATION_KM_FIGSIZE)
    kmf_list = []
    rows = []

    for group in LOCATION_ORDER:
        group_data = plot_df[plot_df["Location"] == group]
        if group_data.empty:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(group_data["time"], group_data["event"], label=group)
        kmf.plot_survival_function(ax=ax, color=LOCATION_COLORS[group], linewidth=2.5, ci_show=False)
        _add_censor_ticks(ax, group_data, LOCATION_COLORS[group], kmf)
        kmf_list.append(kmf)
        rows.append({
            "group": group,
            "n": len(group_data),
            "events": int(group_data["event"].sum()),
        })

    global_lr = multivariate_logrank_test(plot_df["time"], plot_df["Location"], plot_df["event"])
    pairwise = {}
    for i, g1 in enumerate(LOCATION_ORDER):
        for g2 in LOCATION_ORDER[i + 1 :]:
            d1 = plot_df[plot_df["Location"] == g1]
            d2 = plot_df[plot_df["Location"] == g2]
            if len(d1) and len(d2):
                pairwise[f"{g1} vs {g2}"] = logrank_test(
                    d1["time"], d2["time"], d1["event"], d2["event"]
                ).p_value

    p_lines = [f"4-way log-rank: p = {_format_pvalue(global_lr.p_value)}", "Pairwise:"]
    p_lines.extend(f"  {pair}: p = {_format_pvalue(p)}" for pair, p in pairwise.items())
    ax.text(
        0.02,
        0.02,
        "\n".join(p_lines),
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        family="monospace",
    )

    ax.set_xlabel("Time (years)", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.grid(False)
    ax.set_xlim(0, horizon)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", frameon=True, fontsize=10)

    ticks = list(range(0, int(horizon) + 1, 1))
    if int(horizon) not in ticks:
        ticks.append(int(horizon))
    add_at_risk_counts(*kmf_list, ax=ax, xticks=ticks, rows_to_show=["At risk"])

    fig.subplots_adjust(bottom=KM_AT_RISK_BOTTOM)
    out_path = Path(out_path)
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.show(fig)
    plt.close(fig)
    if _REPO_ROOT is not None:
        log_wrote(out_path, _REPO_ROOT)

    summary = pd.DataFrame(rows)
    summary["global_logrank_p"] = global_lr.p_value
    return summary


def plot_km_by_archetype(
    df: pd.DataFrame,
    *,
    time_col: str,
    event_col: str,
    title: str,
    ylabel: str,
    out_path: Path,
    horizon: float = KM_HORIZON_YEARS,
) -> pd.DataFrame:
    plot_df = _prepare_km_frame(df, time_col=time_col, event_col=event_col, horizon=horizon)
    plot_df = plot_df[plot_df["archetype_id"].isin(ARCHETYPE_ORDER)].copy()
    plot_df["archetype_id"] = plot_df["archetype_id"].astype(int)

    fig, ax = plt.subplots(figsize=ARCHETYPE_KM_FIGSIZE)
    kmf_list = []
    rows = []

    for archetype_id in ARCHETYPE_ORDER:
        label = ARCHETYPE_NAME_MAP[archetype_id]
        group_data = plot_df[plot_df["archetype_id"] == archetype_id]
        if group_data.empty:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(
            group_data["time"],
            group_data["event"],
            label=f"{archetype_id}: {label} (n={len(group_data)})",
        )
        kmf.plot_survival_function(
            ax=ax,
            color=ARCHETYPE_COLORS[archetype_id],
            linewidth=2.0,
            ci_show=False,
        )
        _add_censor_ticks(ax, group_data, ARCHETYPE_COLORS[archetype_id], kmf)
        kmf_list.append(kmf)
        rows.append({
            "archetype_id": archetype_id,
            "archetype": label,
            "n": len(group_data),
            "events": int(group_data["event"].sum()),
        })

    global_lr = multivariate_logrank_test(
        plot_df["time"],
        plot_df["archetype_id"].astype(str),
        plot_df["event"],
    )
    p_text = f"Log-rank p = {_format_pvalue(global_lr.p_value)}"
    ax.text(0.02, 0.05, p_text, transform=ax.transAxes, fontsize=10)

    ax.set_xlabel("Time from diagnosis (years)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.8)
    ax.set_xlim(0, horizon)
    ax.set_ylim(0, 1.03)
    ax.legend(loc="upper right", frameon=False, fontsize=8)

    ticks = list(range(0, int(horizon) + 1, 1))
    add_at_risk_counts(*kmf_list, ax=ax, xticks=ticks, rows_to_show=["At risk"])

    fig.subplots_adjust(bottom=KM_AT_RISK_BOTTOM)
    out_path = Path(out_path)
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.show(fig)
    plt.close(fig)
    if _REPO_ROOT is not None:
        log_wrote(out_path, _REPO_ROOT)

    summary = pd.DataFrame(rows)
    summary["global_logrank_p"] = global_lr.p_value
    return summary
