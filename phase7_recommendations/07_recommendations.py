"""
=============================================================
PHASE 7 — BUSINESS RECOMMENDATIONS ENGINE
=============================================================
Synthesizes outputs from all phases into actionable decisions:

  1. Marketing    – which customers to target & how
  2. Pricing      – which products to re-price
  3. Inventory    – stock levels & reorder triggers
  4. Channel      – where to invest/reduce
  5. Risk alerts  – products/customers needing intervention

Outputs a single HTML report + CSV action tables
=============================================================
"""

import pandas as pd
import numpy as np
import os, warnings, pickle
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT_DIR, "data")
STATS_DIR  = os.path.join(DATA_DIR, "stats_outputs")
CLEAN_PATH = os.path.join(DATA_DIR, "superstore_clean.csv")
MODEL_DIR  = os.path.join(ROOT_DIR, "models")
EVAL_DIR   = os.path.join(ROOT_DIR, "evaluation")
REC_DIR    = os.path.join(ROOT_DIR, "recommendations")
os.makedirs(REC_DIR, exist_ok=True)

COLORS = {
    "blue"  : "#3A86FF",
    "red"   : "#FF3A5C",
    "green" : "#06D6A0",
    "amber" : "#FFB703",
    "purple": "#7B5EA7",
    "gray"  : "#8D99AE",
}


def save_fig(fig, name):
    fig.savefig(os.path.join(REC_DIR, f"{name}.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [✓] {name}.png")


# ══════════════════════════════════════════════════════════════
# 1. MARKETING RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════
def marketing_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    print("\n── 1. Marketing Recommendations ────────────────────")
    ref_date = df["order_date"].max()

    cust = (df.groupby("customer_id")
              .agg(
                  customer_name  = ("customer_name", "first"),
                  segment        = ("segment",        "first"),
                  recency        = ("order_date",     lambda x: (ref_date - x.max()).days),
                  frequency      = ("order_id",       "nunique"),
                  monetary       = ("profit",         "sum"),
                  avg_discount   = ("discount",       "mean"),
                  avg_margin     = ("profit_margin",  "mean"),
                  num_channels   = ("channel",        "nunique"),
              ).reset_index())

    # load churn scores if available
    churn_path = f"{MODEL_DIR}/customer_churn_scores.csv"
    if os.path.exists(churn_path):
        churn = pd.read_csv(churn_path)[["customer_id","churn_prob"]]
        cust = cust.merge(churn, on="customer_id", how="left")
    else:
        cust["churn_prob"] = np.where(cust["recency"] > 180, 0.8, 0.2)

    # load rfm segments if available
    rfm_path = f"{MODEL_DIR}/rfm_segments.csv" if os.path.exists(
        f"{MODEL_DIR}/rfm_segments.csv") else os.path.join(STATS_DIR, "rfm_segments.csv")
    if os.path.exists(rfm_path):
        rfm = pd.read_csv(rfm_path)[["customer_id","segment_label"]]
        cust = cust.merge(rfm, on="customer_id", how="left")
    else:
        cust["segment_label"] = np.where(cust["monetary"] > cust["monetary"].quantile(0.75),
                                         "Champions",
                                  np.where(cust["monetary"] > cust["monetary"].quantile(0.50),
                                           "Loyal",
                                  np.where(cust["recency"] > 180, "At Risk", "Potential Loyalist")))

    # assign marketing action
    def assign_action(row):
        if row.get("segment_label") == "Champions":
            return "VIP loyalty rewards + cross-sell premium products"
        elif row.get("segment_label") == "Loyal":
            return "Upsell to higher-margin categories (Technology)"
        elif row.get("churn_prob", 0) > 0.7:
            return "Win-back campaign: targeted 10% coupon"
        elif row.get("segment_label") == "At Risk":
            return "Re-engagement email: show personalized deals"
        elif row.get("recency", 999) > 120:
            return "Dormant reactivation: survey + discount offer"
        else:
            return "Standard newsletter + seasonal promotions"

    cust["recommended_action"] = cust.apply(assign_action, axis=1)
    cust["priority"] = np.where(cust["churn_prob"] > 0.7, "HIGH",
                        np.where(cust["segment_label"] == "Champions", "HIGH",
                        np.where(cust["churn_prob"] > 0.4, "MEDIUM", "LOW")))

    # segment distribution chart
    seg_dist = cust["segment_label"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(seg_dist.index, seg_dist.values,
                  color=list(COLORS.values())[:len(seg_dist)], edgecolor="none")
    for bar, val in zip(bars, seg_dist.values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1,
                str(val), ha="center", fontsize=9)
    ax.set_title("Customer Segment Distribution", fontsize=11, fontweight="bold")
    ax.set_ylabel("# Customers")
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, "01_customer_segments")

    out = cust.sort_values(["priority","monetary"], ascending=[True, False])
    out.to_csv(f"{REC_DIR}/marketing_actions.csv", index=False)
    print(f"  HIGH priority customers : {(out['priority']=='HIGH').sum()}")
    print(f"  MEDIUM priority         : {(out['priority']=='MEDIUM').sum()}")
    print(f"  LOW priority            : {(out['priority']=='LOW').sum()}")
    return out


# ══════════════════════════════════════════════════════════════
# 2. PRICING RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════
def pricing_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    print("\n── 2. Pricing Recommendations ──────────────────────")
    prod = (df.groupby(["sub_category","category"])
              .agg(
                  avg_discount   = ("discount","mean"),
                  avg_margin     = ("profit_margin","mean"),
                  total_profit   = ("profit","sum"),
                  total_revenue  = ("sales","sum"),
                  total_qty      = ("quantity","sum"),
              ).reset_index())

    # load elasticity if available
    elas_path = f"{MODEL_DIR}/price_elasticity.csv"
    if os.path.exists(elas_path):
        elas = pd.read_csv(elas_path)
        prod = prod.merge(elas, on="sub_category", how="left")
    else:
        # fallback estimate
        rng = np.random.default_rng(42)
        prod["elasticity"] = rng.uniform(-2.0, 0.5, len(prod))

    def pricing_action(row):
        margin  = row["avg_margin"]
        elas    = row.get("elasticity", -1)
        disc    = row["avg_discount"]

        if margin < 0:
            return "STOP DISCOUNTING — margin negative; set floor price"
        elif disc > 0.25 and margin < 0.10:
            return "REDUCE discount to ≤10%; margin recovery priority"
        elif elas is not None and elas < -1.5 and margin > 0.20:
            return "Slight price reduction to grow volume (elastic)"
        elif elas is not None and elas > -0.5 and margin > 0.15:
            return "Price increase opportunity — inelastic demand"
        else:
            return "Maintain current pricing; monitor quarterly"

    prod["pricing_action"] = prod.apply(pricing_action, axis=1)
    prod["urgency"] = np.where(prod["avg_margin"] < 0, "CRITICAL",
                       np.where(prod["avg_margin"] < 0.05, "HIGH", "NORMAL"))
    prod.sort_values(["urgency","total_profit"], ascending=[True, True], inplace=True)

    # chart: margin vs discount colored by urgency
    fig, ax = plt.subplots(figsize=(9, 6))
    urgency_colors = {"CRITICAL": COLORS["red"], "HIGH": COLORS["amber"], "NORMAL": COLORS["green"]}
    for urg, grp in prod.groupby("urgency"):
        ax.scatter(grp["avg_discount"]*100, grp["avg_margin"]*100,
                   c=urgency_colors[urg], label=urg,
                   s=grp["total_revenue"]/500, alpha=0.7, edgecolors="none")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(20, color="red", linestyle="--", linewidth=0.8, alpha=0.5, label="20% disc threshold")
    ax.set_xlabel("Avg Discount (%)")
    ax.set_ylabel("Avg Profit Margin (%)")
    ax.set_title("Discount vs Margin by Sub-Category\n(bubble size = revenue)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    save_fig(fig, "02_pricing_matrix")

    prod.to_csv(f"{REC_DIR}/pricing_recommendations.csv", index=False)
    print(f"  CRITICAL pricing issues : {(prod['urgency']=='CRITICAL').sum()}")
    print(f"  HIGH pricing issues     : {(prod['urgency']=='HIGH').sum()}")
    return prod


# ══════════════════════════════════════════════════════════════
# 3. INVENTORY RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════
def inventory_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    print("\n── 3. Inventory Recommendations ────────────────────")
    inv = (df.groupby(["sub_category","category"])
             .agg(
                 monthly_avg_qty = ("quantity","mean"),
                 std_qty         = ("quantity","std"),
                 total_qty       = ("quantity","sum"),
                 total_profit    = ("profit","sum"),
                 avg_margin      = ("profit_margin","mean"),
             ).reset_index())

    inv["safety_stock"] = (1.65 * inv["std_qty"].fillna(0)).round()
    inv["reorder_point"] = (inv["monthly_avg_qty"] + inv["safety_stock"]).round()
    inv["annual_demand"]  = (inv["monthly_avg_qty"] * 12).round()

    # load demand model reorder points if available
    rp_path = f"{MODEL_DIR}/reorder_points.csv"
    if os.path.exists(rp_path):
        rp = pd.read_csv(rp_path, index_col=0).reset_index()
        rp.rename(columns={"index":"sub_category"}, inplace=True, errors="ignore")
        inv = inv.merge(rp[["sub_category","reorder_point"]],
                        on="sub_category", how="left", suffixes=("","_ml"))
        inv["reorder_point"] = inv["reorder_point_ml"].fillna(inv["reorder_point"])
        inv.drop(columns=["reorder_point_ml"], inplace=True)

    def inv_action(row):
        margin = row["avg_margin"]
        if margin < 0:
            return "LIQUIDATE or discontinue — loss-making"
        elif margin > 0.25:
            return "INCREASE stock — high margin, ensure availability"
        elif row["total_qty"] < 50:
            return "LOW VOLUME — reduce SKU count; keep top sellers only"
        else:
            return "MAINTAIN — review reorder point monthly"

    inv["inventory_action"] = inv.apply(inv_action, axis=1)
    inv.sort_values("total_profit", ascending=True, inplace=True)

    # Reorder point chart (top 12 sub-categories by volume)
    top = inv.nlargest(12, "total_qty")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(top))
    ax.bar(x, top["reorder_point"], color=COLORS["blue"],
           alpha=0.7, label="Reorder Point", edgecolor="none")
    ax.bar(x, top["safety_stock"], color=COLORS["amber"],
           alpha=0.7, label="Safety Stock", edgecolor="none")
    ax.set_xticks(list(x))
    ax.set_xticklabels(top["sub_category"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Units")
    ax.set_title("Reorder Points & Safety Stock — Top 12 Sub-Categories",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, "03_inventory_reorder_points")

    inv.to_csv(f"{REC_DIR}/inventory_recommendations.csv", index=False)
    return inv


# ══════════════════════════════════════════════════════════════
# 4. CHANNEL RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════
def channel_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    print("\n── 4. Channel Recommendations ──────────────────────")
    ch = (df.groupby("channel")
            .agg(
                revenue       = ("sales","sum"),
                profit        = ("profit","sum"),
                orders        = ("order_id","nunique"),
                customers     = ("customer_id","nunique"),
                avg_margin    = ("profit_margin","mean"),
                avg_discount  = ("discount","mean"),
            ).reset_index())

    ch["profit_per_order"]    = ch["profit"] / ch["orders"]
    ch["revenue_share_%"]     = ch["revenue"] / ch["revenue"].sum() * 100
    ch["profit_share_%"]      = ch["profit"]  / ch["profit"].sum()  * 100

    def channel_action(row):
        if row["profit_share_%"] > row["revenue_share_%"] * 1.1:
            return "SCALE UP — over-indexing on profit; increase investment"
        elif row["avg_margin"] < 0.05:
            return "REVIEW — low margin; audit discount and cost structure"
        elif row["avg_discount"] > 0.20:
            return "REDUCE DISCOUNTS on this channel; test lower rates"
        else:
            return "MAINTAIN — healthy contribution; optimize AOV"

    ch["channel_action"] = ch.apply(channel_action, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, title, color in zip(
        axes,
        ["revenue", "profit"],
        ["Revenue by Channel", "Profit by Channel"],
        [COLORS["blue"], COLORS["green"]]
    ):
        ax.bar(ch["channel"], ch[col], color=color, edgecolor="none", alpha=0.8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel("$")
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(ax.patches, ch[col]):
            ax.text(bar.get_x() + bar.get_width()/2, val * 1.01,
                    f"${val:,.0f}", ha="center", fontsize=8)
    fig.tight_layout()
    save_fig(fig, "04_channel_performance")

    print(ch[["channel","revenue","profit","avg_margin","channel_action"]].to_string(index=False))
    ch.to_csv(f"{REC_DIR}/channel_recommendations.csv", index=False)
    return ch


# ══════════════════════════════════════════════════════════════
# 5. EXECUTIVE SUMMARY REPORT
# ══════════════════════════════════════════════════════════════
def executive_summary(df, mkt, pricing, inv, ch) -> None:
    print("\n── 5. Executive Summary ────────────────────────────")

    total_rev    = df["sales"].sum()
    total_profit = df["profit"].sum()
    avg_margin   = df["profit_margin"].mean()
    total_orders = df["order_id"].nunique()
    total_cust   = df["customer_id"].nunique()

    # potential profit recovery
    loss_rows   = df[df["profit"] < 0]
    recoverable = abs(loss_rows["profit"].sum())

    high_disc = df[df["discount"] > 0.30]["profit"].sum()
    disc_loss  = abs(min(high_disc, 0))

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Profit Optimization — Executive Summary</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 40px; color: #2c2c2c; }}
  h1   {{ color: #3A86FF; border-bottom: 2px solid #3A86FF; padding-bottom: 8px; }}
  h2   {{ color: #3A86FF; margin-top: 30px; }}
  .kpi-grid {{ display: flex; gap: 20px; flex-wrap: wrap; margin: 20px 0; }}
  .kpi {{ background: #f4f7fb; border-left: 4px solid #3A86FF;
           padding: 12px 20px; border-radius: 4px; min-width: 160px; }}
  .kpi-val {{ font-size: 22px; font-weight: bold; color: #3A86FF; }}
  .kpi-lbl {{ font-size: 12px; color: #666; margin-top: 2px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 10px; font-size: 13px; }}
  th {{ background: #3A86FF; color: white; padding: 8px 12px; text-align: left; }}
  td {{ padding: 7px 12px; border-bottom: 1px solid #eee; }}
  tr:nth-child(even) {{ background: #f9f9f9; }}
  .critical {{ color: #FF3A5C; font-weight: bold; }}
  .good     {{ color: #06D6A0; font-weight: bold; }}
  .warning  {{ color: #FFB703; font-weight: bold; }}
  .section  {{ background: #f9f9f9; padding: 14px 20px; border-radius: 6px;
                margin: 12px 0; border-left: 3px solid #06D6A0; }}
</style>
</head>
<body>
<h1>🏪 Retail Profit Optimization System — Executive Summary</h1>
<p>Generated from Superstore dataset | {len(df):,} transactions | {df['order_date'].min().date()} to {df['order_date'].max().date()}</p>

<h2>📊 Business Snapshot</h2>
<div class="kpi-grid">
  <div class="kpi"><div class="kpi-val">${total_rev:,.0f}</div><div class="kpi-lbl">Total Revenue</div></div>
  <div class="kpi"><div class="kpi-val">${total_profit:,.0f}</div><div class="kpi-lbl">Total Profit</div></div>
  <div class="kpi"><div class="kpi-val">{avg_margin:.1%}</div><div class="kpi-lbl">Avg Profit Margin</div></div>
  <div class="kpi"><div class="kpi-val">{total_orders:,}</div><div class="kpi-lbl">Total Orders</div></div>
  <div class="kpi"><div class="kpi-val">{total_cust:,}</div><div class="kpi-lbl">Unique Customers</div></div>
</div>

<h2>💡 Profit Recovery Opportunities</h2>
<div class="section">
  <b>Recoverable Profit from Loss-Making Lines:</b>
  <span class="critical"> ${recoverable:,.0f}</span><br>
  <b>Profit Drained by Excessive Discounts (>30%):</b>
  <span class="warning"> ${disc_loss:,.0f}</span><br>
  <b>Combined Recovery Potential:</b>
  <span class="good"> ${recoverable + disc_loss:,.0f}</span>
</div>

<h2>🎯 Marketing Actions — Top Priority Customers</h2>
<table>
  <tr><th>Customer</th><th>Segment</th><th>RFM Label</th><th>Churn Risk</th><th>Recommended Action</th></tr>
  {''.join(
    f"<tr><td>{r['customer_name']}</td><td>{r['segment']}</td>"
    f"<td>{r.get('segment_label','—')}</td>"
    f"<td class=\"{'critical' if r.get('churn_prob',0)>0.7 else 'warning' if r.get('churn_prob',0)>0.4 else 'good'}\">"
    f"{r.get('churn_prob',0):.0%}</td>"
    f"<td>{r['recommended_action']}</td></tr>"
    for _, r in mkt[mkt["priority"]=="HIGH"].head(10).iterrows()
  )}
</table>

<h2>💰 Pricing Alerts</h2>
<table>
  <tr><th>Sub-Category</th><th>Avg Margin</th><th>Avg Discount</th><th>Urgency</th><th>Action</th></tr>
  {''.join(
    f"<tr><td>{r['sub_category']}</td>"
    f"<td class=\"{'critical' if r['avg_margin']<0 else 'warning' if r['avg_margin']<0.1 else 'good'}\">"
    f"{r['avg_margin']:.1%}</td>"
    f"<td>{r['avg_discount']:.1%}</td>"
    f"<td class=\"{'critical' if r['urgency']=='CRITICAL' else 'warning'}\">{r['urgency']}</td>"
    f"<td>{r['pricing_action']}</td></tr>"
    for _, r in pricing[pricing["urgency"].isin(["CRITICAL","HIGH"])].head(10).iterrows()
  )}
</table>

<h2>📦 Inventory Alerts</h2>
<table>
  <tr><th>Sub-Category</th><th>Monthly Avg Qty</th><th>Reorder Point</th><th>Safety Stock</th><th>Action</th></tr>
  {''.join(
    f"<tr><td>{r['sub_category']}</td>"
    f"<td>{r['monthly_avg_qty']:.1f}</td>"
    f"<td>{r['reorder_point']:.0f}</td>"
    f"<td>{r['safety_stock']:.0f}</td>"
    f"<td>{r['inventory_action']}</td></tr>"
    for _, r in inv.head(10).iterrows()
  )}
</table>

<h2>📡 Channel Recommendations</h2>
<table>
  <tr><th>Channel</th><th>Revenue</th><th>Profit</th><th>Avg Margin</th><th>Action</th></tr>
  {''.join(
    f"<tr><td>{r['channel']}</td>"
    f"<td>${r['revenue']:,.0f}</td>"
    f"<td>${r['profit']:,.0f}</td>"
    f"<td>{r['avg_margin']:.1%}</td>"
    f"<td>{r['channel_action']}</td></tr>"
    for _, r in ch.iterrows()
  )}
</table>

<h2>📁 Output Files Generated</h2>
<ul>
  <li>data/recommendations/marketing_actions.csv</li>
  <li>data/recommendations/pricing_recommendations.csv</li>
  <li>data/recommendations/inventory_recommendations.csv</li>
  <li>data/recommendations/channel_recommendations.csv</li>
  <li>data/powerbi_exports/  (5 tables for Power BI)</li>
  <li>data/models/           (6 trained ML models)</li>
  <li>data/evaluation/       (ROC, SHAP, lift charts)</li>
</ul>
</body></html>"""

    report_path = f"{REC_DIR}/executive_summary.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  [✓] Executive summary → {report_path}")


# ── main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv(CLEAN_PATH, parse_dates=["order_date","ship_date"])

    mkt     = marketing_recommendations(df)
    pricing = pricing_recommendations(df)
    inv     = inventory_recommendations(df)
    ch      = channel_recommendations(df)
    executive_summary(df, mkt, pricing, inv, ch)

    print(f"\n[✓] All recommendations saved → {REC_DIR}")
