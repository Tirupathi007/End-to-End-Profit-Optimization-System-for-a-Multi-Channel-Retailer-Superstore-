"""
=============================================================
PHASE 6 — POWER BI DASHBOARD SPEC & DAX MEASURES
=============================================================
This file does two things:
  A) Generates the 4 Power BI-ready CSV exports
  B) Prints all DAX measures you copy into Power BI Desktop

DASHBOARD LAYOUT (3 report pages):
  Page 1 : Executive Overview
  Page 2 : Customer Intelligence
  Page 3 : Operations & Risk

=============================================================
"""

import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")

ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT_DIR, "data")
CLEAN_PATH = os.path.join(DATA_DIR, "superstore_clean.csv")
MODEL_DIR  = os.path.join(ROOT_DIR, "models")
PBI_DIR    = os.path.join(ROOT_DIR, "powerbi_exports")
os.makedirs(PBI_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# A. GENERATE POWER BI EXPORT TABLES
# ══════════════════════════════════════════════════════════════
def generate_pbi_tables(df: pd.DataFrame) -> None:
    print("\n── Generating Power BI Export Tables ───────────────")

    # ── 1. fact_sales_pbi (main fact)
    fact = df[[
        "order_id","order_date","order_year","order_month","order_quarter",
        "customer_id","segment","product_id","category","sub_category",
        "region","state","city","channel","ship_mode",
        "sales","quantity","discount","profit","profit_margin",
        "cost","profit_per_unit","is_profitable","ship_days"
    ]].copy()
    fact["order_date"] = fact["order_date"].astype(str)
    fact.to_csv(f"{PBI_DIR}/fact_sales.csv", index=False)
    print(f"  [✓] fact_sales.csv  ({len(fact):,} rows)")

    # ── 2. dim_calendar
    dates = pd.date_range(df["order_date"].min(), df["order_date"].max(), freq="D")
    cal = pd.DataFrame({
        "Date"         : dates,
        "Year"         : dates.year,
        "Quarter"      : dates.quarter,
        "Month"        : dates.month,
        "MonthName"    : dates.strftime("%b"),
        "WeekNum"      : dates.isocalendar().week.astype(int),
        "DayOfWeek"    : dates.dayofweek,
        "DayName"      : dates.strftime("%A"),
        "IsWeekend"    : (dates.dayofweek >= 5).astype(int),
        "QuarterLabel" : dates.year.astype(str) + " Q" + dates.quarter.astype(str)
    })
    cal.to_csv(f"{PBI_DIR}/dim_calendar.csv", index=False)
    print(f"  [✓] dim_calendar.csv  ({len(cal):,} rows)")

    # ── 3. customer_segment_summary (for customer page)
    rfm_path = f"{MODEL_DIR}/rfm_segments.csv"
    churn_path = f"{MODEL_DIR}/customer_churn_scores.csv"

    cust_agg = (df.groupby("customer_id")
                  .agg(
                      customer_name  = ("customer_name","first"),
                      segment        = ("segment","first"),
                      region         = ("region","first"),
                      total_orders   = ("order_id","nunique"),
                      total_revenue  = ("sales","sum"),
                      total_profit   = ("profit","sum"),
                      avg_discount   = ("discount","mean"),
                      avg_margin     = ("profit_margin","mean"),
                      first_order    = ("order_date","min"),
                      last_order     = ("order_date","max"),
                      num_categories = ("category","nunique"),
                      num_channels   = ("channel","nunique"),
                  ).reset_index())
    cust_agg["first_order"] = cust_agg["first_order"].astype(str)
    cust_agg["last_order"]  = cust_agg["last_order"].astype(str)

    if os.path.exists(rfm_path):
        rfm = pd.read_csv(rfm_path)[["customer_id","recency","frequency","monetary","segment_label"]]
        cust_agg = cust_agg.merge(rfm, on="customer_id", how="left")
    if os.path.exists(churn_path):
        churn = pd.read_csv(churn_path)[["customer_id","churn_prob"]]
        cust_agg = cust_agg.merge(churn, on="customer_id", how="left")

    cust_agg.to_csv(f"{PBI_DIR}/dim_customer_enriched.csv", index=False)
    print(f"  [✓] dim_customer_enriched.csv  ({len(cust_agg):,} rows)")

    # ── 4. product_profit_summary
    prod = (df.groupby(["product_id","product_name","category","sub_category"])
              .agg(
                  total_revenue  = ("sales","sum"),
                  total_profit   = ("profit","sum"),
                  total_qty      = ("quantity","sum"),
                  avg_margin     = ("profit_margin","mean"),
                  avg_discount   = ("discount","mean"),
                  times_sold     = ("order_id","count"),
              ).reset_index())
    prod["profit_rank"] = prod["total_profit"].rank(ascending=False).astype(int)

    elas_path = f"{MODEL_DIR}/price_elasticity.csv"
    if os.path.exists(elas_path):
        elas = pd.read_csv(elas_path)
        prod = prod.merge(elas, on="sub_category", how="left")

    prod.to_csv(f"{PBI_DIR}/dim_product_enriched.csv", index=False)
    print(f"  [✓] dim_product_enriched.csv  ({len(prod):,} rows)")

    # ── 5. monthly KPIs for trend visuals
    kpi = (df.groupby(["order_year","order_month","channel","category"])
             .agg(revenue=("sales","sum"), profit=("profit","sum"),
                  orders=("order_id","nunique"), qty=("quantity","sum"))
             .reset_index())
    kpi.to_csv(f"{PBI_DIR}/monthly_kpi.csv", index=False)
    print(f"  [✓] monthly_kpi.csv  ({len(kpi):,} rows)")


# ══════════════════════════════════════════════════════════════
# B. DAX MEASURES
# ══════════════════════════════════════════════════════════════
DAX_MEASURES = """
╔══════════════════════════════════════════════════════════════╗
║  POWER BI DAX MEASURES — Profit Optimization System         ║
║  Paste each measure into Modeling > New Measure             ║
╚══════════════════════════════════════════════════════════════╝

────────────────────────────────────────────────────────────────
PAGE 1 : EXECUTIVE OVERVIEW
────────────────────────────────────────────────────────────────

[Total Revenue] =
    SUM(fact_sales[sales])

[Total Profit] =
    SUM(fact_sales[profit])

[Total Orders] =
    DISTINCTCOUNT(fact_sales[order_id])

[Avg Profit Margin %] =
    DIVIDE(SUM(fact_sales[profit]), SUM(fact_sales[sales]), 0) * 100

[Profit Growth MoM %] =
    VAR CurrentMonthProfit = [Total Profit]
    VAR PrevMonthProfit =
        CALCULATE([Total Profit],
                  DATEADD(dim_calendar[Date], -1, MONTH))
    RETURN
        DIVIDE(CurrentMonthProfit - PrevMonthProfit,
               ABS(PrevMonthProfit), BLANK()) * 100

[Revenue YoY Growth %] =
    VAR ThisYear = [Total Revenue]
    VAR LastYear =
        CALCULATE([Total Revenue],
                  SAMEPERIODLASTYEAR(dim_calendar[Date]))
    RETURN DIVIDE(ThisYear - LastYear, LastYear, BLANK()) * 100

[Running Total Revenue] =
    CALCULATE([Total Revenue],
              FILTER(ALL(dim_calendar),
                     dim_calendar[Date] <= MAX(dim_calendar[Date])))

[Avg Order Value] =
    DIVIDE([Total Revenue], [Total Orders], 0)

[Revenue per Customer] =
    DIVIDE([Total Revenue],
           DISTINCTCOUNT(fact_sales[customer_id]), 0)


────────────────────────────────────────────────────────────────
PAGE 2 : CUSTOMER INTELLIGENCE
────────────────────────────────────────────────────────────────

[Active Customers] =
    CALCULATE(
        DISTINCTCOUNT(fact_sales[customer_id]),
        DATESINPERIOD(dim_calendar[Date],
                      MAX(dim_calendar[Date]), -90, DAY)
    )

[New Customers This Period] =
    COUNTROWS(
        FILTER(
            VALUES(fact_sales[customer_id]),
            CALCULATE(MIN(fact_sales[order_date])) >= MIN(dim_calendar[Date])
        )
    )

[Customer Retention Rate %] =
    VAR ActiveNow =
        CALCULATE(DISTINCTCOUNT(fact_sales[customer_id]),
                  DATESINPERIOD(dim_calendar[Date],
                                MAX(dim_calendar[Date]), -90, DAY))
    VAR ActivePrev =
        CALCULATE(DISTINCTCOUNT(fact_sales[customer_id]),
                  DATESINPERIOD(dim_calendar[Date],
                                MAX(dim_calendar[Date]), -180, DAY),
                  NOT DATESINPERIOD(dim_calendar[Date],
                                    MAX(dim_calendar[Date]), -90, DAY))
    RETURN DIVIDE(ActiveNow, ActiveNow + ActivePrev, 0) * 100

[Avg CLV] =
    AVERAGEX(
        VALUES(dim_customer_enriched[customer_id]),
        CALCULATE(SUM(fact_sales[profit]))
    )

[High Risk Churn Customers] =
    CALCULATE(
        DISTINCTCOUNT(dim_customer_enriched[customer_id]),
        dim_customer_enriched[churn_prob] >= 0.7
    )

[Champions Revenue Share %] =
    VAR ChampRev =
        CALCULATE([Total Revenue],
                  dim_customer_enriched[segment_label] = "Champions")
    RETURN DIVIDE(ChampRev, [Total Revenue], 0) * 100

[RFM Segment Customer Count] =
    COUNTROWS(dim_customer_enriched)


────────────────────────────────────────────────────────────────
PAGE 3 : OPERATIONS & RISK
────────────────────────────────────────────────────────────────

[Avg Ship Days] =
    AVERAGE(fact_sales[ship_days])

[Return Risk Score] =
    -- Uses simulated churn_prob as proxy;
    -- replace with actual return probability when available
    AVERAGE(dim_customer_enriched[churn_prob])

[Loss-Making Orders] =
    CALCULATE(COUNTROWS(fact_sales),
              fact_sales[is_profitable] = 0)

[Loss-Making Order % ] =
    DIVIDE([Loss-Making Orders], [Total Orders], 0) * 100

[Total Discount Given] =
    SUMX(fact_sales, fact_sales[sales] * fact_sales[discount])

[Discount to Revenue Ratio %] =
    DIVIDE([Total Discount Given], [Total Revenue], 0) * 100

[Category Profit Contribution %] =
    DIVIDE([Total Profit],
           CALCULATE([Total Profit], ALL(fact_sales[category])),
           0) * 100

[Inventory Turnover Proxy] =
    DIVIDE(SUM(fact_sales[quantity]),
           CALCULATE(SUM(fact_sales[quantity]),
                     DATEADD(dim_calendar[Date], -1, YEAR)))


────────────────────────────────────────────────────────────────
CONDITIONAL FORMATTING RULES (apply in Format pane)
────────────────────────────────────────────────────────────────

[Margin Color Flag] =
    -- Use as background color rule in table visuals
    -- Green = good,  Amber = ok,  Red = bad
    IF([Avg Profit Margin %] >= 15, "#06D6A0",
       IF([Avg Profit Margin %] >= 5, "#FFB703", "#FF3A5C"))

[KPI Trend Arrow] =
    IF([Profit Growth MoM %] > 0, "▲ " & FORMAT([Profit Growth MoM %],"0.0") & "%",
                                   "▼ " & FORMAT([Profit Growth MoM %],"0.0") & "%")
"""

def print_dax():
    print(DAX_MEASURES)


# ══════════════════════════════════════════════════════════════
# C. DASHBOARD SETUP GUIDE
# ══════════════════════════════════════════════════════════════
DASHBOARD_GUIDE = """
╔══════════════════════════════════════════════════════════════╗
║  POWER BI SETUP STEPS                                       ║
╚══════════════════════════════════════════════════════════════╝

1. Open Power BI Desktop → Get Data → Text/CSV
   Import all 5 files from data/powerbi_exports/:
   • fact_sales.csv
   • dim_calendar.csv
   • dim_customer_enriched.csv
   • dim_product_enriched.csv
   • monthly_kpi.csv

2. DATA MODEL (Manage Relationships):
   fact_sales[order_date]   → dim_calendar[Date]        (many:1)
   fact_sales[customer_id]  → dim_customer_enriched[customer_id] (many:1)
   fact_sales[product_id]   → dim_product_enriched[product_id]   (many:1)

3. PAGE 1 — Executive Overview:
   • KPI Cards : Total Revenue | Total Profit | Avg Margin % | Total Orders
   • Line chart : Running Total Revenue by Month (color by Year)
   • Bar chart  : Profit by Category + Sub-Category (drill-down)
   • Pie chart  : Revenue by Channel
   • Slicer     : Year | Quarter | Region | Segment

4. PAGE 2 — Customer Intelligence:
   • KPI Cards : Active Customers | High Risk Churn | Avg CLV
   • Scatter    : Frequency vs Monetary, colored by segment_label
   • Bar chart  : Top 20 Customers by Total Profit
   • Donut      : RFM segment distribution
   • Table      : Customer name | Segment | Revenue | Churn Prob | Last Order

5. PAGE 3 — Operations & Risk:
   • KPI Cards : Avg Ship Days | Loss-Making Orders % | Discount Ratio %
   • Bar chart  : Profit by Sub-Category (red for negative)
   • Matrix     : Region × Channel revenue/profit
   • Gauge      : Avg Profit Margin vs 15% target
   • Table      : Loss-making products with avg discount

6. THEME:
   File → Options → Report settings → Customize theme
   Primary   : #3A86FF
   Secondary : #06D6A0
   Accent    : #FFB703
   Danger    : #FF3A5C
"""

def print_guide():
    print(DASHBOARD_GUIDE)


# ── main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv(CLEAN_PATH, parse_dates=["order_date","ship_date"])
    generate_pbi_tables(df)
    print_dax()
    print_guide()

    # also write DAX to file for easy copy-paste
    with open(f"{PBI_DIR}/DAX_Measures.txt", "w", encoding="utf-8") as f:
        f.write(DAX_MEASURES)
    with open(f"{PBI_DIR}/Dashboard_Setup_Guide.txt", "w", encoding="utf-8") as f:
        f.write(DASHBOARD_GUIDE)
    print(f"\n[✓] Power BI exports ready → {PBI_DIR}")
