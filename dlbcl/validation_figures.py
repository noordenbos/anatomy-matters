"""Validation-cohort figure builders mirroring discovery Figs 1F, 2F, 3D–O, 4A/4B.

Aesthetics are matched to their discovery counterparts:
  * KM panels   → nb11_km_survival.ipynb (plot_km_by_location / plot_km_by_archetype)
  * GEP / 3D    → nb6 (ANOVA + pairwise DE, class drivers, z-score heatmap)
  * 3E-H / 3I-L → nb6 module-score bar+whisker+jitter
  * 3O          → nb6 GSEA dotplot (one-vs-rest patient-level ranking)

Validation cohort has bulk NanoString GEP (genes x V-aliases), predicted immune
archetypes (nb5 elastic net), and OS only. Panels needing data the cohort lacks
(3M EcoTyper, 3N HLA, 4C/D genomic enrichment) are intentionally not built here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Shared constants (matched to discovery notebooks)
# --------------------------------------------------------------------------- #
ARCHETYPE_NAME_MAP = {1: "low immune", 2: "cytotoxic predominant", 3: "complex immune"}
ARCHETYPE_ORDER = [1, 2, 3]
CLUSTER_ORDER = ["low immune", "cytotoxic predominant", "complex immune"]
SHORT_LABELS = {"low immune": "LI", "cytotoxic predominant": "CP", "complex immune": "CI"}

# nb11 KM colours
LOCATION_ORDER = ["PCNS", "nodal", "bone", "testis"]
LOCATION_COLORS = {"PCNS": "#1a2252", "nodal": "#2fa148", "bone": "#f57f20", "testis": "#d62a28"}
KM_ARCHETYPE_COLORS = {1: "#d73027", 2: "#4575b4", 3: "#1a9850"}

# nb6 GEP module-score colours
CLASS_COLORS = {"low immune": "#d62728", "cytotoxic predominant": "#1f77b4", "complex immune": "#2ca02c"}

# Validation disease_type -> discovery anatomical location vocabulary
DISEASE_TYPE_TO_LOCATION = {
    "Brain": "PCNS",
    "Nodal": "nodal",
    "Bone": "bone",
    "Testis": "testis",
}

KM_HORIZON_YEARS = 5.0
LOCATION_KM_FIGSIZE = (12, 5)  # ~2.2:1 incl. at-risk table
ARCHETYPE_KM_FIGSIZE = (9, 6)  # ~1.8:1 incl. at-risk table
KM_AT_RISK_BOTTOM = 0.28


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_predictions(predictions_csv: Path | str) -> pd.DataFrame:
    """Load nb5 validation predictions (archetype calls merged with clinical meta)."""
    pred = pd.read_csv(predictions_csv)
    pred["patient_alias"] = pred["patient_alias"].astype(str)
    return pred.set_index("patient_alias")


def load_gep(gep_csv: Path | str) -> pd.DataFrame:
    """Normalized validation GEP as genes x patient-alias matrix."""
    gep = pd.read_csv(gep_csv, index_col=0)
    gep.index = gep.index.astype(str).str.strip()
    gep.columns = gep.columns.astype(str).str.strip()
    return gep


def load_from_adata(adata) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load validation inputs from ``adata.uns['validation_cohort']``."""
    from .validation_cohort import cohort_notebook_inputs

    return cohort_notebook_inputs(adata)


def build_survival_table(pred: pd.DataFrame) -> pd.DataFrame:
    """OS survival table: time/event + Location (disease_type) + archetype."""
    surv = pd.DataFrame(index=pred.index)
    surv["time"] = pd.to_numeric(pred["follow_up_time"], errors="coerce")
    surv["event"] = pd.to_numeric(pred["vital_status"], errors="coerce")
    surv["Location"] = pred["disease_type"].map(DISEASE_TYPE_TO_LOCATION)
    surv["archetype_id"] = pd.to_numeric(pred["pred_tumorimmune_archetype_id"], errors="coerce").astype("Int64")
    surv["Archetype"] = surv["archetype_id"].map(ARCHETYPE_NAME_MAP)
    return surv.dropna(subset=["time", "event", "Location", "Archetype"])


# --------------------------------------------------------------------------- #
# Figs 1F / 2F — Kaplan–Meier (OS), aesthetics from nb11
# --------------------------------------------------------------------------- #
def _format_pvalue(p: float) -> str:
    if np.isnan(p):
        return "NA"
    if p < 0.0001:
        return "<0.0001"
    return f"{p:.4f}"


def _prepare_km_frame(df: pd.DataFrame, horizon: float) -> pd.DataFrame:
    out = df.copy()
    out["time"] = pd.to_numeric(out["time"], errors="coerce")
    out["event"] = pd.to_numeric(out["event"], errors="coerce") == 1
    out = out.dropna(subset=["time"])
    out = out[out["time"] > 0].copy()
    out.loc[out["time"] > horizon, "event"] = False
    out.loc[out["time"] > horizon, "time"] = horizon
    return out


def _add_censor_ticks(ax, group_data, color, kmf) -> None:
    censored = group_data[~group_data["event"]]
    for _, row in censored.iterrows():
        survival_prob = float(kmf.predict(row["time"]))
        ax.plot(
            row["time"], survival_prob, marker="|", color=color,
            markersize=12, markeredgewidth=2.5, linestyle="None", zorder=5,
        )


def plot_km_by_location(
    df,
    *,
    title,
    ylabel,
    out_path,
    horizon=KM_HORIZON_YEARS,
    location_order: list[str] | None = None,
    show=True,
):
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter
    from lifelines.plotting import add_at_risk_counts
    from lifelines.statistics import logrank_test, multivariate_logrank_test

    groups = list(location_order) if location_order is not None else list(LOCATION_ORDER)
    plot_df = _prepare_km_frame(df, horizon)
    plot_df = plot_df[plot_df["Location"].isin(groups)].copy()
    active_groups = [g for g in groups if (plot_df["Location"] == g).any()]

    fig, ax = plt.subplots(figsize=LOCATION_KM_FIGSIZE)
    kmf_list, rows = [], []
    for group in groups:
        group_data = plot_df[plot_df["Location"] == group]
        if group_data.empty:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(group_data["time"], group_data["event"], label=group)
        kmf.plot_survival_function(ax=ax, color=LOCATION_COLORS[group], linewidth=2.5, ci_show=False)
        _add_censor_ticks(ax, group_data, LOCATION_COLORS[group], kmf)
        kmf_list.append(kmf)
        rows.append({"group": group, "n": len(group_data), "events": int(group_data["event"].sum())})

    global_lr = multivariate_logrank_test(plot_df["time"], plot_df["Location"], plot_df["event"])
    pairwise = {}
    for i, g1 in enumerate(active_groups):
        for g2 in active_groups[i + 1:]:
            d1 = plot_df[plot_df["Location"] == g1]
            d2 = plot_df[plot_df["Location"] == g2]
            if len(d1) and len(d2):
                pairwise[f"{g1} vs {g2}"] = logrank_test(
                    d1["time"], d2["time"], d1["event"], d2["event"]
                ).p_value

    n_groups = len(active_groups)
    p_lines = [f"{n_groups}-way log-rank: p = {_format_pvalue(global_lr.p_value)}", "Pairwise:"]
    p_lines.extend(f"  {pair}: p = {_format_pvalue(p)}" for pair, p in pairwise.items())
    ax.text(
        0.02, 0.02, "\n".join(p_lines), transform=ax.transAxes, fontsize=9,
        ha="left", va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9), family="monospace",
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
    if show:
        plt.show(fig)
    plt.close(fig)

    summary = pd.DataFrame(rows)
    summary["global_logrank_p"] = global_lr.p_value
    return summary


def plot_km_by_archetype(df, *, title, ylabel, out_path, horizon=KM_HORIZON_YEARS, show=True):
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter
    from lifelines.plotting import add_at_risk_counts
    from lifelines.statistics import multivariate_logrank_test

    plot_df = _prepare_km_frame(df, horizon)
    plot_df = plot_df[plot_df["archetype_id"].isin(ARCHETYPE_ORDER)].copy()
    plot_df["archetype_id"] = plot_df["archetype_id"].astype(int)

    fig, ax = plt.subplots(figsize=ARCHETYPE_KM_FIGSIZE)
    kmf_list, rows = [], []
    for archetype_id in ARCHETYPE_ORDER:
        label = ARCHETYPE_NAME_MAP[archetype_id]
        group_data = plot_df[plot_df["archetype_id"] == archetype_id]
        if group_data.empty:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(group_data["time"], group_data["event"], label=f"{archetype_id}: {label} (n={len(group_data)})")
        kmf.plot_survival_function(ax=ax, color=KM_ARCHETYPE_COLORS[archetype_id], linewidth=2.0, ci_show=False)
        _add_censor_ticks(ax, group_data, KM_ARCHETYPE_COLORS[archetype_id], kmf)
        kmf_list.append(kmf)
        rows.append({
            "archetype_id": archetype_id, "archetype": label,
            "n": len(group_data), "events": int(group_data["event"].sum()),
        })

    global_lr = multivariate_logrank_test(plot_df["time"], plot_df["archetype_id"].astype(str), plot_df["event"])
    ax.text(0.02, 0.05, f"Log-rank p = {_format_pvalue(global_lr.p_value)}", transform=ax.transAxes, fontsize=10)

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
    if show:
        plt.show(fig)
    plt.close(fig)

    summary = pd.DataFrame(rows)
    summary["global_logrank_p"] = global_lr.p_value
    return summary


# --------------------------------------------------------------------------- #
# Fig 3D — differential expression by archetype + class-driver heatmap (nb6)
# --------------------------------------------------------------------------- #
def _align_expr_labels(expr: pd.DataFrame, archetype_label: pd.Series):
    common = pd.Index(expr.columns).intersection(archetype_label.dropna().index)
    X = expr.loc[:, common].T.copy()
    y = archetype_label.loc[common].copy()
    keep = y.isin(CLUSTER_ORDER)
    X = X.loc[keep.values].apply(pd.to_numeric, errors="coerce")
    y = y.loc[keep]
    X = X.loc[:, X.notna().any(axis=0)]
    X = X.loc[:, X.var(axis=0, ddof=1).fillna(0) > 0]
    keep_genes = X.notna().mean(axis=0) >= 0.8
    X = X.loc[:, keep_genes]
    X = X.apply(lambda s: s.fillna(s.median()), axis=0)
    return X, y


def compute_archetype_de(expr: pd.DataFrame, archetype_label: pd.Series):
    """Global ANOVA + pairwise t-tests + UP/DOWN class drivers (mirrors nb6)."""
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    X, y = _align_expr_labels(expr, archetype_label)
    cluster_order = [c for c in CLUSTER_ORDER if c in set(y)]

    # Global 3-class ANOVA
    global_rows = []
    for gene in X.columns:
        vals = [X.loc[y == cls, gene].values for cls in cluster_order]
        if any(len(v) < 2 for v in vals):
            continue
        f_stat, p_val = stats.f_oneway(*vals)
        means = {f"mean_{cls}": float(np.mean(X.loc[y == cls, gene])) for cls in cluster_order}
        global_rows.append({
            "gene": gene, "F": f_stat, "p": p_val, **means,
            "range_mean": max(means.values()) - min(means.values()),
            "std_mean": float(np.std(list(means.values()), ddof=1)),
        })
    global_de = pd.DataFrame(global_rows)
    global_de["q"] = multipletests(global_de["p"], method="fdr_bh")[1]
    global_de = global_de.sort_values(["q", "p", "range_mean"], ascending=[True, True, False])

    # Pairwise DE
    def run_pairwise(class_a, class_b):
        Xa, Xb = X.loc[y == class_a], X.loc[y == class_b]
        rows = []
        for gene in X.columns:
            va, vb = Xa[gene].values, Xb[gene].values
            t_stat, p_val = stats.ttest_ind(va, vb, equal_var=False, nan_policy="omit")
            diff = np.nanmean(va) - np.nanmean(vb)
            rows.append({"gene": gene, "class_a": class_a, "class_b": class_b, "diff": diff, "p": p_val})
        df = pd.DataFrame(rows)
        df["q"] = multipletests(df["p"].fillna(1), method="fdr_bh")[1]
        return df

    pairwise = pd.concat(
        [run_pairwise(a, b) for i, a in enumerate(cluster_order) for b in cluster_order[i + 1:]],
        ignore_index=True,
    )

    # Class drivers (UP/DOWN vs both others)
    GLOBAL_Q_TH, DRIVER_Q_TH, DRIVER_DIFF_TH = 0.10, 0.05, 0.25

    def signed_effect(target, comparator):
        direct = pairwise[(pairwise["class_a"] == target) & (pairwise["class_b"] == comparator)]
        if not direct.empty:
            out = direct[["gene", "diff", "q"]].copy()
            out.columns = ["gene", f"diff_{target}_{comparator}", f"q_{target}_{comparator}"]
            return out
        rev = pairwise[(pairwise["class_a"] == comparator) & (pairwise["class_b"] == target)].copy()
        rev[f"diff_{target}_{comparator}"] = -rev["diff"]
        rev = rev.rename(columns={"q": f"q_{target}_{comparator}"})
        return rev[["gene", f"diff_{target}_{comparator}", f"q_{target}_{comparator}"]]

    driver_tables = []
    global_keep = global_de[["gene", "F", "p", "q", "range_mean", "std_mean"]].rename(
        columns={"p": "global_p", "q": "global_q"}
    )
    for target in cluster_order:
        others = [c for c in cluster_order if c != target]
        tmp = global_keep.copy()
        for comp in others:
            tmp = tmp.merge(signed_effect(target, comp), on="gene", how="left")
        diff_cols = [f"diff_{target}_{c}" for c in others]
        q_cols = [f"q_{target}_{c}" for c in others]
        tmp["target_class"] = target
        tmp["min_diff_vs_others"] = tmp[diff_cols].min(axis=1)
        tmp["max_diff_vs_others"] = tmp[diff_cols].max(axis=1)
        tmp["max_pairwise_q"] = tmp[q_cols].max(axis=1)
        tmp["driver_direction"] = np.select(
            [
                (tmp["min_diff_vs_others"] >= DRIVER_DIFF_TH) & (tmp["max_pairwise_q"] < DRIVER_Q_TH) & (tmp["global_q"] < GLOBAL_Q_TH),
                (tmp["max_diff_vs_others"] <= -DRIVER_DIFF_TH) & (tmp["max_pairwise_q"] < DRIVER_Q_TH) & (tmp["global_q"] < GLOBAL_Q_TH),
            ],
            ["UP", "DOWN"], default="not_driver",
        )
        tmp["driver_score"] = np.where(
            tmp["driver_direction"] == "UP",
            tmp["min_diff_vs_others"] * -np.log10(tmp["max_pairwise_q"].clip(lower=1e-300)),
            np.where(
                tmp["driver_direction"] == "DOWN",
                tmp["max_diff_vs_others"].abs() * -np.log10(tmp["max_pairwise_q"].clip(lower=1e-300)),
                np.nan,
            ),
        )
        driver_tables.append(tmp)
    class_drivers = pd.concat(driver_tables, ignore_index=True).sort_values(
        ["target_class", "driver_direction", "driver_score"], ascending=[True, True, False]
    )
    return X, y, cluster_order, global_de, pairwise, class_drivers


def plot_top_driver_heatmap(X, y, cluster_order, class_drivers, *, out_path, top_per_class_direction=50, show=True):
    """Fig 3D — z-scored class-mean heatmap of top class-driving genes (nb6)."""
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import leaves_list, linkage
    from sklearn.preprocessing import StandardScaler

    genes = []
    for cls in cluster_order:
        for direction in ["UP", "DOWN"]:
            g = (
                class_drivers[(class_drivers["target_class"] == cls) & (class_drivers["driver_direction"] == direction)]
                .sort_values("driver_score", ascending=False)
                .head(top_per_class_direction)["gene"].tolist()
            )
            genes.extend(g)
    genes = list(dict.fromkeys(genes))
    if not genes:
        return None, pd.DataFrame()

    mean_expr = pd.DataFrame({cls: X.loc[y == cls, genes].mean(axis=0) for cls in cluster_order}).T
    zmat = pd.DataFrame(
        StandardScaler().fit_transform(mean_expr), index=mean_expr.index, columns=mean_expr.columns
    )
    if zmat.shape[1] > 1:
        gene_order = zmat.columns[leaves_list(linkage(zmat.T, method="average"))]
    else:
        gene_order = zmat.columns
    zmat = zmat.loc[cluster_order, gene_order]

    fig, ax = plt.subplots(figsize=(max(8, 0.22 * zmat.shape[1]), 3.2))
    im = ax.imshow(zmat.values, aspect="auto", interpolation="nearest", cmap="coolwarm")
    ax.set_yticks(np.arange(len(zmat.index)))
    ax.set_yticklabels(zmat.index)
    ax.set_xticks(np.arange(len(zmat.columns)))
    ax.set_xticklabels(zmat.columns, rotation=90, fontsize=7)
    ax.set_title("Top class-driving genes (validation)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Gene z-score across classes")
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show(fig)
    plt.close(fig)
    return fig, zmat


# --------------------------------------------------------------------------- #
# Figs 3E-H / 3I-L — module scores (nb6 gene sets)
# --------------------------------------------------------------------------- #
IMMUNE_THELPER_MODULES = {
    "tcell_identity": ["CD3D", "CD3E", "CD3G", "CD2", "TRAC", "TRBC1", "TRBC2", "LCK", "LAT", "ZAP70"],
    "cytotoxicity": ["NKG7", "PRF1", "GZMB", "GZMA", "GZMH", "GNLY", "IFNG", "CX3CR1", "KLRD1", "KLRC2", "KLRK1", "FGFBP2"],
    "ifng_activation": ["IFNG", "STAT1", "IRF1", "CXCL9", "CXCL10", "GBP1", "GBP5", "IDO1", "HLA-DRA", "HLA-DRB1", "CD274"],
    "terminal_exhaustion": ["PDCD1", "HAVCR2", "LAG3", "TIGIT", "CTLA4", "TOX", "TOX2", "ENTPD1", "CXCL13", "LAYN"],
    "progenitor_exhausted_memory": ["TCF7", "IL7R", "SLAMF6", "LEF1", "CXCR5", "CCR7"],
    "senescence_effector_like": ["KLRG1", "B3GAT1", "CX3CR1", "GZMH", "FGFBP2", "PRF1", "NKG7", "TBX21", "EOMES"],
    "Th1_IFNg": ["TBX21", "IFNG", "STAT1", "STAT4", "IL12RB1", "IL12RB2", "CXCR3", "CCR5", "CXCL9", "CXCL10", "CXCL11", "GBP1", "GBP5"],
    "Th17_IL17": ["RORC", "IL17A", "IL17F", "IL23R", "CCR6", "KLRB1", "STAT3", "RORA", "IL21", "IL22", "CCL20"],
    "Th2_IL4": ["GATA3", "IL4", "IL5", "IL13", "CCR4", "CCR8", "STAT6", "MAF", "IL1RL1", "PTGDR2"],
    "Tfh_GC_helper": ["CXCR5", "BCL6", "PDCD1", "ICOS", "SH2D1A", "IL21", "CD40LG", "CXCL13", "MAF", "TOX2"],
    "Treg_suppressive": ["FOXP3", "IL2RA", "CTLA4", "IKZF2", "TIGIT", "TNFRSF18", "CCR8", "ENTPD1", "BATF", "LAYN"],
}

THELPER_MODULES = {
    "Th1_IFNg": IMMUNE_THELPER_MODULES["Th1_IFNg"],
    "Th17_IL17": IMMUNE_THELPER_MODULES["Th17_IL17"],
    "Th2_IL4": IMMUNE_THELPER_MODULES["Th2_IL4"],
    "Tfh_GC_helper": IMMUNE_THELPER_MODULES["Tfh_GC_helper"],
    "Treg_suppressive": IMMUNE_THELPER_MODULES["Treg_suppressive"],
    "Naive_memory_CD4": ["TCF7", "LEF1", "IL7R", "CCR7", "SELL", "LTB", "MAL"],
}


def _zscore_gene_rows(df):
    mu = df.mean(axis=1)
    sd = df.std(axis=1, ddof=0).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0).fillna(0)


def _format_p(p):
    if pd.isna(p):
        return "NA"
    if p < 1e-4:
        return "<0.0001"
    if p < 1e-3:
        return "<0.001"
    if p < 1e-2:
        return "<0.01"
    return f"{p:.2f}"


def compute_module_scores(expr, archetype_label, modules, contrast_defs, inflammatory=True):
    """Per-patient z-scored module scores + derived contrasts (nb6)."""
    common = expr.columns.intersection(archetype_label.dropna().index)
    e = expr.loc[:, common].copy()
    lab = archetype_label.loc[common]
    keep = lab.isin(CLUSTER_ORDER)
    e = e.loc[:, keep.values]
    lab = lab.loc[keep]

    score_rows, presence_rows = [], []
    for module, genes in modules.items():
        present = [g for g in genes if g in e.index]
        presence_rows.append({
            "module": module, "n_requested": len(genes), "n_present": len(present),
            "present_genes": ", ".join(present), "missing_genes": ", ".join(g for g in genes if g not in e.index),
        })
        if len(present) < 2:
            continue
        scores = _zscore_gene_rows(e.loc[present]).mean(axis=0)
        for patient, score in scores.items():
            score_rows.append({"patient_id": patient, "cluster_name": lab.loc[patient], "module": module, "score": score})

    score_long = pd.DataFrame(score_rows)
    presence_df = pd.DataFrame(presence_rows)
    score_wide = score_long.pivot_table(
        index=["patient_id", "cluster_name"], columns="module", values="score", aggfunc="first"
    ).reset_index()

    if inflammatory and {"Th1_IFNg", "Th17_IL17", "Th2_IL4"}.issubset(score_wide.columns):
        score_wide["Inflammatory_Th1Th17_minus_Th2"] = (
            score_wide[["Th1_IFNg", "Th17_IL17"]].mean(axis=1) - score_wide["Th2_IL4"]
        )
    for new_col, (a, b) in contrast_defs.items():
        if new_col in score_wide.columns:
            continue
        if a in score_wide.columns and b in score_wide.columns:
            score_wide[new_col] = score_wide[a] - score_wide[b]

    contrast_cols = [c for c in score_wide.columns if c not in score_long["module"].unique().tolist() + ["patient_id", "cluster_name"]]
    contrast_long = score_wide.melt(
        id_vars=["patient_id", "cluster_name"], value_vars=contrast_cols, var_name="module", value_name="score"
    )
    plot_long = pd.concat(
        [score_long[["patient_id", "cluster_name", "module", "score"]], contrast_long], ignore_index=True
    )
    return plot_long, score_wide, presence_df


def module_kruskal_stats(plot_long):
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    rows = []
    for module, sub in plot_long.groupby("module"):
        groups = [sub.loc[sub["cluster_name"] == cls, "score"].dropna().values for cls in CLUSTER_ORDER]
        if all(len(g) >= 2 for g in groups):
            H, p = stats.kruskal(*groups)
        else:
            H, p = np.nan, np.nan
        rows.append({"module": module, "test": "Kruskal-Wallis", "statistic": H, "p": p})
    df = pd.DataFrame(rows)
    df["q"] = multipletests(df["p"].fillna(1), method="fdr_bh")[1]
    return df


def plot_module_scores(
    plot_long,
    stats_df,
    modules_to_plot,
    *,
    out_path,
    suptitle,
    figsize_per=(4.5, 4.0),
    seed=42,
    jitter_width=0.10,
    box_width=0.55,
    show=True,
):
    import matplotlib.pyplot as plt

    modules_to_plot = [m for m in modules_to_plot if m in plot_long["module"].unique()]
    ncols = 3
    nrows = int(np.ceil(len(modules_to_plot) / ncols))
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows), sharey=False)
    axes = np.array(axes).reshape(-1)

    for ax, module in zip(axes, modules_to_plot):
        sub = plot_long[plot_long["module"] == module]
        x = np.arange(len(CLUSTER_ORDER))
        box_data = [
            sub.loc[sub["cluster_name"] == cls, "score"].dropna().values
            for cls in CLUSTER_ORDER
        ]
        bp = ax.boxplot(
            box_data,
            positions=x,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.2},
            whiskerprops={"color": "black", "linewidth": 0.8},
            capprops={"color": "black", "linewidth": 0.8},
            boxprops={"linewidth": 0.8, "edgecolor": "black"},
        )
        for patch, cls in zip(bp["boxes"], CLUSTER_ORDER):
            patch.set_facecolor(CLASS_COLORS[cls])
            patch.set_alpha(0.65)

        for i, cls in enumerate(CLUSTER_ORDER):
            vals = sub.loc[sub["cluster_name"] == cls, "score"].dropna().values
            jitter = rng.normal(0, jitter_width, size=len(vals))
            ax.scatter(
                np.full(len(vals), i) + jitter,
                vals,
                s=34,
                color=CLASS_COLORS[cls],
                edgecolor="black",
                linewidth=0.4,
                alpha=0.9,
                zorder=3,
            )
        p_row = stats_df.loc[stats_df["module"] == module]
        if len(p_row):
            title = f"{module}\np {_format_p(p_row['p'].iloc[0])} | FDR {_format_p(p_row['q'].iloc[0])}"
        else:
            title = module
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT_LABELS[c] for c in CLUSTER_ORDER], fontsize=9)
        ax.set_ylabel("Module score")
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    for ax in axes[len(modules_to_plot):]:
        ax.axis("off")

    fig.suptitle(suptitle, fontsize=15, y=1.01)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig 3O — GSEA dotplot (one-vs-rest patient-level ranking; nb6 aesthetics)
# --------------------------------------------------------------------------- #
def _make_rank_vector_one_vs_rest(X, y, positive_class, seed=42):
    from scipy.stats import ttest_ind

    mask_pos = (y == positive_class).values
    A, B = X.iloc[mask_pos, :], X.iloc[~mask_pos, :]
    diff = A.mean(axis=0) - B.mean(axis=0)
    t_stat, _ = ttest_ind(A.values, B.values, axis=0, equal_var=False, nan_policy="omit")
    res = pd.DataFrame({"gene": X.columns.astype(str), "stat": t_stat, "diff": diff.values})
    res = res.replace([np.inf, -np.inf], np.nan).dropna(subset=["gene", "stat"])
    res = res.groupby("gene", as_index=False).agg(stat=("stat", "mean"), diff=("diff", "mean"))
    res["rank_score"] = res["stat"] + 1e-9 * res["diff"]
    rng = np.random.default_rng(seed)
    res["rank_score"] = res["rank_score"] + rng.normal(0, 1e-10, size=len(res))
    return res.sort_values("rank_score", ascending=False).reset_index(drop=True)


def run_archetype_gsea(X, y, cluster_order, *, genesets="MSigDB_Hallmark_2020", permutations=1000,
                       min_size=8, max_size=300, seed=42):
    import gseapy as gp

    def standardize(res2d):
        res = res2d.copy()
        if res.empty:
            return pd.DataFrame(columns=["Term", "NES", "fdr"])
        if "Term" not in res.columns:
            res = res.reset_index().rename(columns={"index": "Term"})
        res = res.rename(columns={"FDR q-val": "fdr", "NOM p-val": "pval"})
        if "fdr" not in res.columns:
            res["fdr"] = np.nan
        for c in ["NES", "ES", "pval", "fdr"]:
            if c in res.columns:
                res[c] = pd.to_numeric(res[c], errors="coerce")
        return res[[c for c in ["Term", "NES", "ES", "pval", "fdr"] if c in res.columns]]

    all_rows = []
    for cls in cluster_order:
        rank_df = _make_rank_vector_one_vs_rest(X, y, cls, seed=seed)
        rnk = rank_df.set_index("gene")["rank_score"].sort_values(ascending=False)
        g = gp.prerank(rnk=rnk, gene_sets=genesets, permutation_num=permutations,
                       min_size=min_size, max_size=max_size, outdir=None, seed=seed, verbose=False)
        res = standardize(g.res2d)
        if res.empty:
            continue
        res["cluster_name"] = cls
        all_rows.append(res)
    if not all_rows:
        raise ValueError("No GSEA results returned.")
    return pd.concat(all_rows, ignore_index=True)


def plot_gsea_dotplot(gsea_long, cluster_order, *, out_path, genesets="MSigDB_Hallmark_2020",
                      top_m=10, fdr_th=0.25, size_cap=14.0, size_scale=35.0,
                      nonsignificant_size=18.0, cmap="PuOr_r", show=True):
    import matplotlib.pyplot as plt

    col_labels = [SHORT_LABELS[c] for c in cluster_order]
    nes_mat = gsea_long.pivot_table(index="Term", columns="cluster_name", values="NES", aggfunc="first").reindex(columns=cluster_order)
    fdr_mat = gsea_long.pivot_table(index="Term", columns="cluster_name", values="fdr", aggfunc="first").reindex(columns=cluster_order)

    diff_metric = nes_mat.max(axis=1) - nes_mat.min(axis=1)
    valid = nes_mat.notna().sum(axis=1) >= 2
    keep_terms = valid & ((fdr_mat.min(axis=1) <= 0.25) | (nes_mat.abs().max(axis=1) >= 1.5))
    sel = diff_metric.loc[keep_terms].sort_values(ascending=False)
    if sel.empty:
        raise ValueError("No GSEA pathways pass selection.")
    terms_keep = sel.head(top_m).index.tolist()
    nes_sel, fdr_sel = nes_mat.loc[terms_keep], fdr_mat.loc[terms_keep]

    plot_df = (
        nes_sel.stack(dropna=False).rename("NES").to_frame()
        .join(fdr_sel.stack(dropna=False).rename("fdr"), how="left")
        .reset_index().rename(columns={"level_0": "Term", "level_1": "cluster_name"})
    )
    neglog = np.clip(-np.log10(np.clip(plot_df["fdr"].values, 1e-300, 1.0)), 0, size_cap)
    finite_nes = plot_df["NES"].notna()
    sig = finite_nes & (plot_df["fdr"] <= fdr_th)
    plot_df["size"] = np.where(finite_nes, nonsignificant_size, 0.0)
    plot_df.loc[sig, "size"] = np.maximum(
        nonsignificant_size,
        neglog[sig.values] * size_scale,
    )
    x_map = {cls: i for i, cls in enumerate(cluster_order)}
    y_terms = list(nes_sel.index)
    y_map = {t: i for i, t in enumerate(y_terms)}
    plot_df["x"] = plot_df["cluster_name"].map(x_map)
    plot_df["y"] = plot_df["Term"].map(y_map)

    fig_w = max(7, 1.25 * len(cluster_order))
    fig_h = max(6, 0.40 * len(y_terms) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sc = ax.scatter(plot_df["x"], plot_df["y"], c=plot_df["NES"], s=plot_df["size"],
                    cmap=cmap, vmin=-2, vmax=2, edgecolors="none")
    ax.set_xlim(-0.5, len(cluster_order) - 0.5)
    ax.set_xticks(range(len(cluster_order)))
    ax.set_xticklabels(col_labels, fontsize=12)
    ax.set_yticks(range(len(y_terms)))
    ax.set_yticklabels(y_terms, fontsize=14)
    ax.invert_yaxis()
    ax.set_xlabel("tumorimmune_archetype_id (predicted)", fontsize=13)
    ax.set_ylabel("Pathway", fontsize=13)
    ax.set_title(f"GSEA dotplot ({genesets}) — one-vs-rest, validation", fontsize=14)
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="x", linestyle=":", linewidth=0.6, color="0.7")
    ax.grid(False, axis="y")
    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(1.5)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_ticks([-2, 0, 2])
    cbar.set_label("NES", fontsize=14)
    plt.tight_layout()

    handles = [
        ax.scatter([], [], s=nonsignificant_size, c="white", edgecolors="black", linewidths=0.3)
    ]
    labels = [f"FDR > {fdr_th:g}"]
    for nv in [1, 2, 3, 4]:
        if nv > size_cap:
            continue
        handles.append(ax.scatter([], [], s=nv * size_scale, c="white", edgecolors="black", linewidths=0.3))
        labels.append(f"-log10(FDR)={nv:g}")
    ax.legend(handles, labels, title="Significance", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)

    out_path = Path(out_path)
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show(fig)
    plt.close(fig)
    return nes_sel, fdr_sel


# --------------------------------------------------------------------------- #
# Fig 4A / 4B — integration metadata (location + archetype + Hans COO)
# --------------------------------------------------------------------------- #
def build_integration_metadata_validation(
    pred: pd.DataFrame,
    case_classifications: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Patient-level table for integration panels (discovery ``case_classifications`` columns)."""
    from .integration_figures import COLUMN_ALIASES

    if case_classifications is not None:
        meta = case_classifications.copy()
        meta.index = meta.index.astype(str)
        meta.index.name = "patient_id"
        for old_col, new_col in COLUMN_ALIASES.items():
            if old_col in meta.columns and new_col not in meta.columns:
                meta[new_col] = meta[old_col]
        return meta[meta["Location"].notna()].copy()

    meta = pd.DataFrame(index=pred.index.astype(str))
    meta.index.name = "patient_id"
    meta["Location"] = pred["disease_type"].map(DISEASE_TYPE_TO_LOCATION)
    meta["tumorimmune_archetype_id"] = pd.to_numeric(pred["pred_tumorimmune_archetype_id"], errors="coerce").astype("Int64")
    meta["tumorimmune_archetype"] = meta["tumorimmune_archetype_id"].map(ARCHETYPE_NAME_MAP)
    coo_map = {"GCB": "GCB", "Non-GCB": "ABC"}
    meta["COO_NanoString"] = pred["coo_hans"].map(coo_map)
    return meta[meta["Location"].notna()].copy()


# --------------------------------------------------------------------------- #
# Fig 3M — EcoTyper B-cell states by archetype (nb6 aesthetics)
# --------------------------------------------------------------------------- #
ECOTYPER_BSTATE_ORDER = ["S01", "S02", "S03", "S04", "S05"]
ECOTYPER_BSTATE_COLORS = {
    "S01": "#f4a21d",
    "S02": "#ffd84d",
    "S03": "#7ac943",
    "S04": "#3fa9f5",
    "S05": "#2f4aa8",
}
ECOTYPER_BSTATE_LABELS = {
    "S01": "S1",
    "S02": "S2",
    "S03": "S3",
    "S04": "S4",
    "S05": "S5",
}


def plot_ecotyper_bcell_by_archetype(
    bstate: pd.DataFrame,
    archetype: pd.Series,
    *,
    out_dir: Path,
    out_stem: str = "ecotyper_b_state_by_archetype",
    title: str = "Distribution of EcoTyper B-cell states across immune archetypes (validation)",
    show: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stacked barplot of dominant EcoTyper B states × predicted archetype."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scipy.stats import chi2_contingency

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    archetype_col = "tumorimmune_archetype"
    archetype = archetype.rename(archetype_col) if archetype.name != archetype_col else archetype
    archetype = archetype.reindex(bstate.index.astype(str))
    plot_df = bstate[["Dominant_B_cell_state"]].join(archetype, how="inner").dropna()
    plot_df = plot_df[plot_df["Dominant_B_cell_state"].isin(ECOTYPER_BSTATE_ORDER)]

    count_tab = pd.crosstab(plot_df[archetype_col], plot_df["Dominant_B_cell_state"])
    count_tab = count_tab.reindex(
        index=[x for x in CLUSTER_ORDER if x in count_tab.index],
        columns=ECOTYPER_BSTATE_ORDER,
        fill_value=0,
    )
    prop_tab = count_tab.div(count_tab.sum(axis=1), axis=0).fillna(0)

    chi2, p_val, dof, expected = chi2_contingency(count_tab)
    stats_df = pd.DataFrame([{
        "test": "Chi-square",
        "comparison": "immune archetype x EcoTyper B-cell state",
        "chi2": chi2,
        "dof": dof,
        "p": p_val,
        "p_label": _format_pvalue(p_val),
        "min_expected_count": float(pd.DataFrame(expected, index=count_tab.index, columns=count_tab.columns).min().min()),
    }])

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    x = np.arange(len(prop_tab.index))
    bottom = np.zeros(len(prop_tab))
    for state in ECOTYPER_BSTATE_ORDER:
        if state not in prop_tab.columns:
            continue
        vals = prop_tab[state].to_numpy()
        ax.bar(x, vals, bottom=bottom, color=ECOTYPER_BSTATE_COLORS[state], width=0.72, edgecolor="white", linewidth=0.4)
        for i, (xi, val) in enumerate(zip(x, vals)):
            if val >= 0.08:
                ax.text(xi, bottom[i] + val / 2, f"{int(count_tab.loc[prop_tab.index[i], state])}", ha="center", va="center", fontsize=8, color="black")
        bottom += vals

    ax.text(
        0.5, 1.02,
        f"Global association: χ²({int(dof)}) = {chi2:.2f}, p = {_format_pvalue(p_val)}",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=10, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(prop_tab.index)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Fraction of patients", fontsize=12)
    ax.set_xlabel("")
    ax.set_title(title, fontsize=14, pad=28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend_handles = [
        Patch(facecolor=ECOTYPER_BSTATE_COLORS[s], edgecolor="none", label=ECOTYPER_BSTATE_LABELS[s])
        for s in ECOTYPER_BSTATE_ORDER
    ]
    ax.legend(handles=legend_handles, title="EcoTyper B state", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    for ext in (".svg", ".png", ".pdf"):
        fig.savefig(out_dir / f"{out_stem}{ext}", bbox_inches="tight", dpi=300 if ext == ".png" else None)
    if show:
        plt.show(fig)
    plt.close(fig)

    count_tab.to_csv(out_dir / f"{out_stem}_counts.csv")
    stats_df.to_csv(out_dir / f"{out_stem}_chisq_stats.csv", index=False)
    return count_tab, prop_tab, stats_df
