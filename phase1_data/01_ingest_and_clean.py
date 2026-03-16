"""
=============================================================
PHASE 1 — DATA SOURCES & INGESTION
=============================================================
Superstore CSV → cleaned, validated DataFrame
Columns used:
  Row ID, Order ID, Order Date, Ship Date, Ship Mode,
  Customer ID, Customer Name, Segment, Country, City, State,
  Postal Code, Region, Product ID, Category, Sub-Category,
  Product Name, Sales, Quantity, Discount, Profit
=============================================================
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────
RAW_PATH    = "C:/Users/HP/Downloads/profit_optimizer_complete/profit_optimizer/data/Superstore.csv"
CLEAN_PATH  = "C:/Users/HP/Downloads/profit_optimizer_complete/profit_optimizer/data/superstore_clean.csv"

# ── 1. load ──────────────────────────────────────────────────
def load_raw(path: str) -> pd.DataFrame:
    """Load CSV with fallback encodings."""
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"[✓] Loaded  {len(df):,} rows  |  encoding={enc}")
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode file with known encodings.")


# ── 2. inspect ───────────────────────────────────────────────
def inspect(df: pd.DataFrame) -> None:
    print("\n── Schema ──────────────────────────────────────────")
    print(df.dtypes)
    print("\n── Nulls ───────────────────────────────────────────")
    nulls = df.isnull().sum()
    print(nulls[nulls > 0] if nulls.any() else "  No nulls found.")
    print("\n── Shape ──", df.shape)
    print("── Sample ─\n", df.head(3).to_string())


# ── 3. clean ─────────────────────────────────────────────────
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # normalise column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    # parse dates
    for col in ["order_date", "ship_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")

    # derived date fields
    df["order_year"]     = df["order_date"].dt.year
    df["order_month"]    = df["order_date"].dt.month
    df["order_quarter"]  = df["order_date"].dt.quarter
    df["order_dayofweek"]= df["order_date"].dt.dayofweek   # 0=Mon
    df["ship_days"]      = (df["ship_date"] - df["order_date"]).dt.days

    # numeric corrections
    df["sales"]    = pd.to_numeric(df["sales"],    errors="coerce").fillna(0)
    df["profit"]   = pd.to_numeric(df["profit"],   errors="coerce").fillna(0)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(1)
    df["discount"] = pd.to_numeric(df["discount"], errors="coerce").fillna(0).clip(0, 1)

    # derived business metrics
    df["revenue"]         = df["sales"]                          # alias for clarity
    df["profit_margin"]   = np.where(df["sales"] != 0,
                                     df["profit"] / df["sales"], 0)
    df["cost"]            = df["sales"] - df["profit"]
    df["profit_per_unit"] = np.where(df["quantity"] != 0,
                                     df["profit"] / df["quantity"], 0)
    df["is_profitable"]   = (df["profit"] > 0).astype(int)

    # simulate channel (not in raw data — add for multi-channel story)
    rng = np.random.default_rng(42)
    df["channel"] = rng.choice(
        ["website", "mobile_app", "store"],
        size=len(df),
        p=[0.45, 0.30, 0.25]
    )

    # drop exact duplicates
    before = len(df)
    df.drop_duplicates(subset=["order_id", "product_id"], keep="first", inplace=True)
    print(f"[✓] Dropped {before - len(df)} duplicate rows")

    # remove negative quantities
    df = df[df["quantity"] > 0].reset_index(drop=True)

    print(f"[✓] Clean DataFrame: {df.shape}")
    return df


# ── 4. validate ───────────────────────────────────────────────
def validate(df: pd.DataFrame) -> None:
    print("\n── Validation ──────────────────────────────────────")
    assert df["sales"].min() >= 0,    "Negative sales found!"
    assert df["discount"].between(0, 1).all(), "Discount out of [0,1]!"
    assert df["quantity"].min() > 0,  "Non-positive quantity found!"
    assert df["order_date"].notna().all(), "Null order dates!"
    print("  [✓] All validation checks passed.")


# ── 5. summary stats ──────────────────────────────────────────
def summary(df: pd.DataFrame) -> None:
    print("\n── Business Summary ────────────────────────────────")
    print(f"  Total Revenue : ${df['sales'].sum():,.2f}")
    print(f"  Total Profit  : ${df['profit'].sum():,.2f}")
    print(f"  Avg Margin    : {df['profit_margin'].mean():.2%}")
    print(f"  Unique Customers: {df['customer_id'].nunique():,}")
    print(f"  Unique Products : {df['product_id'].nunique():,}")
    print(f"  Date Range    : {df['order_date'].min().date()} → {df['order_date'].max().date()}")
    print(f"\n  By Channel:\n{df.groupby('channel')[['sales','profit']].sum().round(2)}")
    print(f"\n  By Segment:\n{df.groupby('segment')[['sales','profit']].sum().round(2)}")
    print(f"\n  By Category:\n{df.groupby('category')[['sales','profit','profit_margin']].mean().round(4)}")


# ── main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    df_raw   = load_raw(RAW_PATH)
    inspect(df_raw)
    df_clean = clean(df_raw)
    validate(df_clean)
    summary(df_clean)
    df_clean.to_csv(CLEAN_PATH, index=False)
    print(f"\n[✓] Saved → {CLEAN_PATH}")
