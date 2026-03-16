"""
=============================================================
PHASE 2 — SQL DATA WAREHOUSE & ANALYTICAL QUERIES
=============================================================
• Builds a SQLite star-schema from the clean CSV
• Runs 10 analytical SQL queries covering:
    - Profit by channel / segment / category / region
    - RFM scoring
    - Discount impact on profit
    - Product-level profitability
    - Monthly revenue trend
=============================================================
"""

import sqlite3
import pandas as pd
import numpy as np
import os

CLEAN_PATH = "C:/Users/HP/Downloads/profit_optimizer_complete/profit_optimizer/data/superstore_clean.csv"
DB_PATH    = "C:/Users/HP/Downloads/profit_optimizer_complete/profit_optimizer/data/superstore_dw.db"


# ══════════════════════════════════════════════════════════════
# 1. BUILD STAR SCHEMA
# ══════════════════════════════════════════════════════════════

DDL_STATEMENTS = """
-- ── Dimension tables ──────────────────────────────────────

CREATE TABLE IF NOT EXISTS dim_customer (
    customer_id   TEXT PRIMARY KEY,
    customer_name TEXT,
    segment       TEXT
);

CREATE TABLE IF NOT EXISTS dim_product (
    product_id   TEXT PRIMARY KEY,
    product_name TEXT,
    category     TEXT,
    sub_category TEXT
);

CREATE TABLE IF NOT EXISTS dim_geography (
    geo_key     INTEGER PRIMARY KEY AUTOINCREMENT,
    city        TEXT,
    state       TEXT,
    region      TEXT,
    postal_code TEXT
);

CREATE TABLE IF NOT EXISTS dim_date (
    date_key    TEXT PRIMARY KEY,   -- YYYY-MM-DD
    year        INTEGER,
    quarter     INTEGER,
    month       INTEGER,
    day         INTEGER,
    dayofweek   INTEGER,
    is_weekend  INTEGER
);

-- ── Fact table ────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS fact_sales (
    row_id          INTEGER PRIMARY KEY,
    order_id        TEXT,
    order_date      TEXT,
    ship_date       TEXT,
    ship_mode       TEXT,
    ship_days       INTEGER,
    customer_id     TEXT REFERENCES dim_customer(customer_id),
    product_id      TEXT REFERENCES dim_product(product_id),
    geo_key         INTEGER REFERENCES dim_geography(geo_key),
    channel         TEXT,
    sales           REAL,
    quantity        INTEGER,
    discount        REAL,
    profit          REAL,
    profit_margin   REAL,
    cost            REAL,
    profit_per_unit REAL,
    is_profitable   INTEGER
);
"""


def build_star_schema(df: pd.DataFrame, conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.executescript(DDL_STATEMENTS)

    # dim_customer
    cust = df[["customer_id","customer_name","segment"]].drop_duplicates("customer_id")
    cust.to_sql("dim_customer", conn, if_exists="replace", index=False)

    # dim_product
    prod = df[["product_id","product_name","category","sub_category"]].drop_duplicates("product_id")
    prod.to_sql("dim_product", conn, if_exists="replace", index=False)

    # dim_geography
    geo = (df[["city","state","region","postal_code"]]
           .drop_duplicates()
           .reset_index(drop=True))
    geo.index += 1
    geo.index.name = "geo_key"
    geo.to_sql("dim_geography", conn, if_exists="replace", index=True)

    # merge geo_key back
    df = df.merge(geo.reset_index(), on=["city","state","region","postal_code"], how="left")

    # dim_date
    dates = pd.to_datetime(df["order_date"].unique())
    dim_date = pd.DataFrame({
        "date_key" : dates.strftime("%Y-%m-%d"),
        "year"     : dates.year,
        "quarter"  : dates.quarter,
        "month"    : dates.month,
        "day"      : dates.day,
        "dayofweek": dates.dayofweek,
        "is_weekend": (dates.dayofweek >= 5).astype(int)
    }).drop_duplicates("date_key")
    dim_date.to_sql("dim_date", conn, if_exists="replace", index=False)

    # fact_sales
    fact_cols = [
        "row_id","order_id","order_date","ship_date","ship_mode","ship_days",
        "customer_id","product_id","geo_key","channel",
        "sales","quantity","discount","profit",
        "profit_margin","cost","profit_per_unit","is_profitable"
    ]
    df["order_date"] = df["order_date"].astype(str)
    df["ship_date"]  = df["ship_date"].astype(str)

    # row_id fallback
    if "row_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "row_id"})

    df[fact_cols].to_sql("fact_sales", conn, if_exists="replace", index=False)

    conn.commit()
    print(f"[✓] Star schema built — {len(df):,} fact rows loaded")


# ══════════════════════════════════════════════════════════════
# 2. ANALYTICAL SQL QUERIES
# ══════════════════════════════════════════════════════════════

QUERIES = {

"Q1_profit_by_channel": """
-- Total revenue, profit, and margin by sales channel
SELECT
    channel,
    COUNT(*)                          AS orders,
    ROUND(SUM(sales),2)               AS total_revenue,
    ROUND(SUM(profit),2)              AS total_profit,
    ROUND(AVG(profit_margin)*100, 2)  AS avg_margin_pct
FROM fact_sales
GROUP BY channel
ORDER BY total_profit DESC;
""",

"Q2_rfm_scores": """
-- RFM (Recency, Frequency, Monetary) per customer
-- Using max order_date as reference point
WITH ref AS (SELECT MAX(order_date) AS max_date FROM fact_sales),
rfm_raw AS (
    SELECT
        f.customer_id,
        c.segment,
        CAST(julianday((SELECT max_date FROM ref))
             - julianday(MAX(f.order_date)) AS INT)   AS recency_days,
        COUNT(DISTINCT f.order_id)                    AS frequency,
        ROUND(SUM(f.sales),2)                         AS monetary
    FROM fact_sales f
    JOIN dim_customer c USING (customer_id)
    GROUP BY f.customer_id
),
rfm_scored AS (
    SELECT *,
        NTILE(5) OVER (ORDER BY recency_days DESC) AS r_score,
        NTILE(5) OVER (ORDER BY frequency)         AS f_score,
        NTILE(5) OVER (ORDER BY monetary)          AS m_score
    FROM rfm_raw
)
SELECT *,
       (r_score + f_score + m_score)              AS rfm_total,
       CASE
           WHEN r_score >= 4 AND f_score >= 4 THEN 'Champions'
           WHEN r_score >= 3 AND f_score >= 3 THEN 'Loyal'
           WHEN r_score >= 3 AND f_score <= 2 THEN 'Potential Loyalist'
           WHEN r_score <= 2 AND f_score >= 3 THEN 'At Risk'
           ELSE 'Need Attention'
       END                                        AS rfm_segment
FROM rfm_scored
ORDER BY rfm_total DESC
LIMIT 20;
""",

"Q3_profit_by_category": """
-- Profitability by Category × Sub-Category
SELECT
    p.category,
    p.sub_category,
    COUNT(*)                            AS line_items,
    ROUND(SUM(f.sales),2)               AS revenue,
    ROUND(SUM(f.profit),2)              AS profit,
    ROUND(AVG(f.profit_margin)*100,2)   AS avg_margin_pct,
    ROUND(AVG(f.discount)*100,2)        AS avg_discount_pct,
    SUM(f.is_profitable)                AS profitable_orders
FROM fact_sales f
JOIN dim_product p USING (product_id)
GROUP BY p.category, p.sub_category
ORDER BY profit DESC;
""",

"Q4_discount_impact": """
-- How discount bands affect profit margin
SELECT
    CASE
        WHEN discount = 0           THEN '0%  No discount'
        WHEN discount <= 0.10       THEN '1–10%'
        WHEN discount <= 0.20       THEN '11–20%'
        WHEN discount <= 0.30       THEN '21–30%'
        ELSE '> 30%'
    END                                   AS discount_band,
    COUNT(*)                              AS orders,
    ROUND(AVG(profit_margin)*100, 2)      AS avg_margin_pct,
    ROUND(SUM(profit),2)                  AS total_profit,
    ROUND(SUM(sales),2)                   AS total_revenue
FROM fact_sales
GROUP BY discount_band
ORDER BY avg_margin_pct DESC;
""",

"Q5_monthly_revenue_trend": """
-- Monthly revenue and profit trend
SELECT
    d.year,
    d.month,
    d.quarter,
    ROUND(SUM(f.sales),2)  AS revenue,
    ROUND(SUM(f.profit),2) AS profit,
    ROUND(AVG(f.profit_margin)*100,2) AS margin_pct
FROM fact_sales f
JOIN dim_date d ON f.order_date = d.date_key
GROUP BY d.year, d.month
ORDER BY d.year, d.month;
""",

"Q6_top_profitable_products": """
-- Top 15 most profitable products
SELECT
    p.product_name,
    p.category,
    p.sub_category,
    ROUND(SUM(f.sales),2)               AS total_revenue,
    ROUND(SUM(f.profit),2)              AS total_profit,
    ROUND(AVG(f.profit_margin)*100,2)   AS avg_margin_pct,
    COUNT(*)                            AS times_sold
FROM fact_sales f
JOIN dim_product p USING (product_id)
GROUP BY p.product_id
ORDER BY total_profit DESC
LIMIT 15;
""",

"Q7_loss_making_products": """
-- Products draining profit
SELECT
    p.product_name,
    p.category,
    ROUND(SUM(f.profit),2)  AS total_profit,
    ROUND(AVG(f.discount)*100,2) AS avg_discount_pct,
    COUNT(*)                AS times_sold
FROM fact_sales f
JOIN dim_product p USING (product_id)
GROUP BY p.product_id
HAVING total_profit < 0
ORDER BY total_profit ASC
LIMIT 15;
""",

"Q8_customer_clv": """
-- Customer Lifetime Value proxy
SELECT
    c.customer_id,
    c.customer_name,
    c.segment,
    COUNT(DISTINCT f.order_id)          AS total_orders,
    ROUND(SUM(f.sales),2)               AS total_revenue,
    ROUND(SUM(f.profit),2)              AS total_profit,
    ROUND(AVG(f.profit_margin)*100,2)   AS avg_margin_pct,
    MIN(f.order_date)                   AS first_order,
    MAX(f.order_date)                   AS last_order
FROM fact_sales f
JOIN dim_customer c USING (customer_id)
GROUP BY c.customer_id
ORDER BY total_profit DESC
LIMIT 20;
""",

"Q9_region_performance": """
-- Region × Channel profit matrix
SELECT
    g.region,
    f.channel,
    ROUND(SUM(f.sales),2)    AS revenue,
    ROUND(SUM(f.profit),2)   AS profit,
    COUNT(DISTINCT f.order_id) AS orders
FROM fact_sales f
JOIN dim_geography g USING (geo_key)
GROUP BY g.region, f.channel
ORDER BY g.region, profit DESC;
""",

"Q10_shipping_efficiency": """
-- Ship mode vs profit and ship days
SELECT
    ship_mode,
    ROUND(AVG(ship_days),1)            AS avg_ship_days,
    COUNT(*)                           AS orders,
    ROUND(SUM(profit),2)               AS total_profit,
    ROUND(AVG(profit_margin)*100,2)    AS avg_margin_pct
FROM fact_sales
GROUP BY ship_mode
ORDER BY avg_ship_days;
"""

}


def run_queries(conn: sqlite3.Connection) -> dict:
    results = {}
    print("\n── Query Results ───────────────────────────────────")
    for name, sql in QUERIES.items():
        try:
            df = pd.read_sql_query(sql, conn)
            results[name] = df
            print(f"\n  [{name}]")
            print(df.to_string(index=False))
        except Exception as e:
            print(f"  [!] {name} failed: {e}")
    return results


# ── main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv(CLEAN_PATH)
    conn = sqlite3.connect(DB_PATH)
    build_star_schema(df, conn)
    results = run_queries(conn)
    conn.close()
    print(f"\n[✓] Database saved → {DB_PATH}")
