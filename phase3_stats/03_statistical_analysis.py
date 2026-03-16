"""
=============================================================
PHASE 3 — STATISTICAL ANALYSIS
=============================================================
• Correlation analysis  – what drives profit?
• ANOVA test            – do channels/segments differ significantly?
• A/B test simulation   – discount vs no-discount
• Customer segmentation – RFM-based k-means
• Distribution analysis – profit margin, sales outliers
• Saves outputs to ../data/stats_outputs/
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ── paths ────────────────────────────────────────────────────
CLEAN_PATH  = "C:/Users/HP/Downloads/profit_optimizer_complete/profit_optimizer/data/superstore_clean.csv"
OUTPUT_DIR  = "C:/Users/HP/Downloads/profit_optimizer_complete/profit_optimizer/data/stats_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    "blue"  : "#3A86FF",
    "red"   : "#FF3A5C",
    "green" : "#06D6A0",
    "amber" : "#FFB703",
    "purple": "#7B5EA7",
    "gray"  : "#8D99AE",
}
PALETTE = list(COLORS.values())


# ══════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════
def save(fig, name: str) -> None:
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [✓] Saved {path}")


# ══════════════════════════════════════════════════════════════
# 1. CORRELATION ANALYSIS
# ══════════════════════════════════════════════════════════════
def correlation_analysis(df: pd.DataFrame) -> None:
    print("\n── 1. Correlation Analysis ─────────────────────────")
    num_cols = ["sales","quantity","discount","profit","profit_margin",
                "ship_days","cost","profit_per_unit"]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(num_cols, fontsize=9)
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="black" if abs(corr.iloc[i, j]) < 0.6 else "white")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    save(fig, "01_correlation_matrix")

    # key insight
    profit_corr = corr["profit"].drop("profit").sort_values(ascending=False)
    print("  Top drivers of profit:")
    print(profit_corr.to_string())


# ══════════════════════════════════════════════════════════════
# 2. ANOVA — channel differences
# ══════════════════════════════════════════════════════════════
def anova_channel(df: pd.DataFrame) -> None:
    print("\n── 2. ANOVA — Profit Margin by Channel ─────────────")
    groups = [g["profit_margin"].dropna() for _, g in df.groupby("channel")]
    F, p = stats.f_oneway(*groups)
    print(f"  F-statistic = {F:.4f}  |  p-value = {p:.6f}")
    print(f"  {'Significant difference across channels (p<0.05)' if p < 0.05 else 'No significant difference'}")

    fig, ax = plt.subplots(figsize=(8, 5))
    channels = df["channel"].unique()
    data     = [df[df["channel"] == c]["profit_margin"] for c in channels]
    bp = ax.boxplot(data, patch_artist=True, labels=channels, notch=False)
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_ylabel("Profit Margin")
    ax.set_title(f"Profit Margin by Channel  (ANOVA F={F:.2f}, p={p:.4f})",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "02_anova_channel")


def anova_segment(df: pd.DataFrame) -> None:
    print("\n── 2b. ANOVA — Profit Margin by Segment ────────────")
    groups = [g["profit_margin"].dropna() for _, g in df.groupby("segment")]
    F, p = stats.f_oneway(*groups)
    print(f"  F-statistic = {F:.4f}  |  p-value = {p:.6f}")

    agg = df.groupby("segment")["profit_margin"].mean().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(agg.index, agg.values * 100,
                   color=PALETTE[:len(agg)], edgecolor="none")
    for bar, val in zip(bars, agg.values * 100):
        ax.text(val + 0.2, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("Avg Profit Margin (%)")
    ax.set_title("Avg Profit Margin by Customer Segment", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    save(fig, "02b_anova_segment")


# ══════════════════════════════════════════════════════════════
# 3. A/B TEST — Discount vs No Discount
# ══════════════════════════════════════════════════════════════
def ab_test_discount(df: pd.DataFrame) -> None:
    print("\n── 3. A/B Test — Discount Impact on Profit ─────────")
    no_disc = df[df["discount"] == 0]["profit_margin"]
    with_disc = df[df["discount"] > 0]["profit_margin"]

    t, p = stats.ttest_ind(no_disc, with_disc, equal_var=False)
    print(f"  No-discount   n={len(no_disc):,}  mean={no_disc.mean():.4f}")
    print(f"  With-discount n={len(with_disc):,}  mean={with_disc.mean():.4f}")
    print(f"  Welch t={t:.4f}  p={p:.6f}")
    print(f"  → {'Discount significantly hurts margin (p<0.05)' if p < 0.05 else 'No significant effect'}")

    # discount band breakdown
    bins   = [-0.001, 0.001, 0.10, 0.20, 0.30, 1.0]
    labels = ["0%", "1–10%", "11–20%", "21–30%", ">30%"]
    df["disc_band"] = pd.cut(df["discount"], bins=bins, labels=labels)
    agg = df.groupby("disc_band", observed=True)["profit_margin"].mean() * 100

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(agg.index, agg.values,
                  color=[COLORS["green"] if v > 0 else COLORS["red"] for v in agg.values],
                  edgecolor="none")
    for bar, val in zip(bars, agg.values):
        ax.text(bar.get_x() + bar.get_width()/2, val + (0.3 if val >= 0 else -1.2),
                f"{val:.1f}%", ha="center", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Discount Band")
    ax.set_ylabel("Avg Profit Margin (%)")
    ax.set_title("Impact of Discount on Profit Margin", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "03_ab_discount_impact")


# ══════════════════════════════════════════════════════════════
# 4. CUSTOMER SEGMENTATION (RFM + K-MEANS)
# ══════════════════════════════════════════════════════════════
def customer_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    print("\n── 4. Customer Segmentation (RFM + K-Means) ────────")
    ref_date = df["order_date"].max()

    rfm = (df.groupby("customer_id")
             .agg(
                 recency   = ("order_date", lambda x: (ref_date - x.max()).days),
                 frequency = ("order_id",   "nunique"),
                 monetary  = ("profit",     "sum"),
             )
             .reset_index())

    # scale
    scaler   = StandardScaler()
    rfm_s    = scaler.fit_transform(rfm[["recency","frequency","monetary"]])

    # elbow → 4 clusters works well for Superstore
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm["cluster"] = km.fit_predict(rfm_s)

    # label clusters by monetary mean
    cluster_means = rfm.groupby("cluster")["monetary"].mean().sort_values(ascending=False)
    label_map = {
        cluster_means.index[0]: "Champions",
        cluster_means.index[1]: "Loyal",
        cluster_means.index[2]: "At Risk",
        cluster_means.index[3]: "Lost",
    }
    rfm["segment_label"] = rfm["cluster"].map(label_map)

    print(rfm.groupby("segment_label")[["recency","frequency","monetary"]].mean().round(2))

    # scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    seg_colors = {
        "Champions": COLORS["green"],
        "Loyal"    : COLORS["blue"],
        "At Risk"  : COLORS["amber"],
        "Lost"     : COLORS["red"],
    }
    for seg, grp in rfm.groupby("segment_label"):
        ax.scatter(grp["frequency"], grp["monetary"],
                   c=seg_colors[seg], label=seg, alpha=0.6, s=40, edgecolors="none")
    ax.set_xlabel("Frequency (orders)")
    ax.set_ylabel("Monetary (total profit $)")
    ax.set_title("Customer Segmentation — RFM Clusters", fontsize=12, fontweight="bold")
    ax.legend(title="Segment")
    ax.grid(alpha=0.3)
    save(fig, "04_customer_rfm_clusters")

    # save RFM table
    rfm.to_csv(os.path.join(OUTPUT_DIR, "rfm_segments.csv"), index=False)
    return rfm


# ══════════════════════════════════════════════════════════════
# 5. PROFIT MARGIN DISTRIBUTION
# ══════════════════════════════════════════════════════════════
def profit_distribution(df: pd.DataFrame) -> None:
    print("\n── 5. Profit Margin Distribution ───────────────────")
    margin = df["profit_margin"].clip(-1, 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # histogram
    axes[0].hist(margin, bins=60, color=COLORS["blue"], edgecolor="none", alpha=0.8)
    axes[0].axvline(margin.mean(),  color="red",   linestyle="--", label=f"Mean={margin.mean():.2f}")
    axes[0].axvline(margin.median(),color="green", linestyle="--", label=f"Median={margin.median():.2f}")
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set_xlabel("Profit Margin")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Profit Margin Distribution", fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # by category
    cat_margin = df.groupby("category")["profit_margin"].apply(list)
    axes[1].boxplot(cat_margin.values, labels=cat_margin.index,
                    patch_artist=True,
                    boxprops=dict(facecolor=COLORS["blue"], alpha=0.6))
    axes[1].axhline(0, color="red", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("Profit Margin")
    axes[1].set_title("Margin Distribution by Category", fontsize=11, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save(fig, "05_profit_margin_distribution")

    # skewness & kurtosis
    print(f"  Skewness  : {stats.skew(margin):.4f}")
    print(f"  Kurtosis  : {stats.kurtosis(margin):.4f}")
    _, p_norm = stats.normaltest(margin)
    print(f"  Normality test p={p_norm:.4e} → {'NOT normal' if p_norm < 0.05 else 'Normal'}")


# ══════════════════════════════════════════════════════════════
# 6. MONTHLY TREND ANALYSIS
# ══════════════════════════════════════════════════════════════
def trend_analysis(df: pd.DataFrame) -> None:
    print("\n── 6. Monthly Revenue & Profit Trend ───────────────")
    monthly = (df.groupby(["order_year","order_month"])
                 .agg(revenue=("sales","sum"), profit=("profit","sum"))
                 .reset_index())
    monthly["period"] = pd.to_datetime(
        monthly["order_year"].astype(str) + "-" + monthly["order_month"].astype(str).str.zfill(2))
    monthly.sort_values("period", inplace=True)

    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax2 = ax1.twinx()
    ax1.fill_between(monthly["period"], monthly["revenue"],
                     color=COLORS["blue"], alpha=0.25, label="Revenue")
    ax1.plot(monthly["period"], monthly["revenue"],
             color=COLORS["blue"], linewidth=1.5)
    ax2.plot(monthly["period"], monthly["profit"],
             color=COLORS["green"], linewidth=2, label="Profit")
    ax1.set_ylabel("Revenue ($)", color=COLORS["blue"])
    ax2.set_ylabel("Profit ($)",  color=COLORS["green"])
    ax1.set_xlabel("Month")
    ax1.set_title("Monthly Revenue & Profit Trend", fontsize=12, fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "06_monthly_trend")


# ── main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv(CLEAN_PATH, parse_dates=["order_date","ship_date"])

    correlation_analysis(df)
    anova_channel(df)
    anova_segment(df)
    ab_test_discount(df)
    rfm_df = customer_segmentation(df)
    rfm_df.to_csv(os.path.join(OUTPUT_DIR, "rfm_segments.csv"), index=False)
    profit_distribution(df)
    trend_analysis(df)

    print(f"\n[✓] All statistical outputs saved → {OUTPUT_DIR}")
