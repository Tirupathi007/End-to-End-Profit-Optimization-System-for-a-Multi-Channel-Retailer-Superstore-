"""
=============================================================
PHASE 4 — ML / DL MODELS
=============================================================
Model 1 : Revenue Forecasting          (Prophet / LSTM fallback)
Model 2 : Customer Churn Prediction    (XGBoost)
Model 3 : Returns Risk Scoring         (Random Forest)
Model 4 : Product Recommender          (Collaborative Filtering)
Model 5 : Price Elasticity             (Regression)
Model 6 : Demand / Inventory Forecast  (XGBoost time-series)

All models save their artefacts to ../data/models/
=============================================================
"""

import pandas as pd
import numpy as np
import os, warnings, pickle
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection   import train_test_split, cross_val_score
from sklearn.preprocessing     import StandardScaler, LabelEncoder
from sklearn.ensemble          import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model      import LinearRegression, Ridge
from sklearn.metrics           import (
    classification_report, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from xgboost import XGBClassifier, XGBRegressor

# ── paths ────────────────────────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT_DIR, "data")
CLEAN_PATH  = os.path.join(DATA_DIR, "superstore_clean.csv")
MODEL_DIR   = os.path.join(ROOT_DIR, "models")
STATS_DIR   = os.path.join(DATA_DIR, "stats_outputs")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── shared helpers ────────────────────────────────────────────
def save_fig(fig, name: str) -> None:
    fig.savefig(os.path.join(MODEL_DIR, f"{name}.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def save_pkl(obj, name: str) -> None:
    with open(os.path.join(MODEL_DIR, f"{name}.pkl"), "wb") as f:
        pickle.dump(obj, f)
    print(f"  [✓] Saved {name}.pkl")

def regression_metrics(y_true, y_pred, label="") -> None:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"  {label}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")


# ══════════════════════════════════════════════════════════════
# MODEL 1 — REVENUE FORECASTING (Prophet)
# ══════════════════════════════════════════════════════════════
def model1_revenue_forecast(df: pd.DataFrame) -> None:
    print("\n── Model 1: Revenue Forecasting ────────────────────")
    monthly = (df.groupby(["order_year","order_month"])["sales"]
                 .sum().reset_index())
    monthly["ds"] = pd.to_datetime(
        monthly["order_year"].astype(str) + "-" + monthly["order_month"].astype(str).str.zfill(2))
    monthly = monthly.rename(columns={"sales": "y"})[["ds","y"]].sort_values("ds")

    try:
        from prophet import Prophet
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, seasonality_mode="multiplicative")
        m.fit(monthly)
        future  = m.make_future_dataframe(periods=12, freq="MS")
        forecast = m.predict(future)

        fig1 = m.plot(forecast)
        fig1.suptitle("Prophet Revenue Forecast — Next 12 Months",
                      fontsize=11, fontweight="bold")
        save_fig(fig1, "m1_prophet_forecast")

        fig2 = m.plot_components(forecast)
        save_fig(fig2, "m1_prophet_components")

        save_pkl(m, "model1_prophet")
        print("  [Prophet] Training done.")

    except ImportError:
        print("  [Prophet not installed] → Using XGBoost time-series fallback")
        # lag features
        monthly["lag1"]  = monthly["y"].shift(1)
        monthly["lag3"]  = monthly["y"].shift(3)
        monthly["lag12"] = monthly["y"].shift(12)
        monthly["month_num"] = monthly["ds"].dt.month
        monthly.dropna(inplace=True)

        X = monthly[["lag1","lag3","lag12","month_num"]]
        y = monthly["y"]
        split = int(len(X) * 0.8)
        Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]

        xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
        xgb.fit(Xtr, ytr)
        preds = xgb.predict(Xte)
        regression_metrics(yte, preds, "XGB Revenue Forecast")
        save_pkl(xgb, "model1_xgb_revenue")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(len(ytr)), ytr, label="Train", color="#3A86FF", alpha=0.5)
        ax.plot(range(len(ytr), len(ytr)+len(yte)), yte,
                label="Actual", color="#06D6A0", linewidth=2)
        ax.plot(range(len(ytr), len(ytr)+len(preds)), preds,
                label="Predicted", color="#FF3A5C", linestyle="--", linewidth=2)
        ax.set_title("XGB Revenue Forecast (Fallback)", fontsize=11, fontweight="bold")
        ax.legend(); ax.grid(alpha=0.3)
        save_fig(fig, "m1_xgb_forecast")


# ══════════════════════════════════════════════════════════════
# MODEL 2 — CUSTOMER CHURN PREDICTION (XGBoost)
# ══════════════════════════════════════════════════════════════
def model2_churn_prediction(df: pd.DataFrame) -> None:
    print("\n── Model 2: Churn Prediction ───────────────────────")
    ref_date = df["order_date"].max()

    cust = (df.groupby("customer_id")
              .agg(
                  recency        = ("order_date", lambda x: (ref_date - x.max()).days),
                  frequency      = ("order_id",   "nunique"),
                  monetary       = ("sales",       "sum"),
                  total_profit   = ("profit",      "sum"),
                  avg_discount   = ("discount",    "mean"),
                  avg_margin     = ("profit_margin","mean"),
                  num_categories = ("category",    "nunique"),
                  num_channels   = ("channel",     "nunique"),
              ).reset_index())

    # label: "churned" = no order in last 180 days
    cust["churned"] = (cust["recency"] > 180).astype(int)
    print(f"  Churn rate: {cust['churned'].mean():.2%}")

    feat_cols = ["recency","frequency","monetary","total_profit",
                 "avg_discount","avg_margin","num_categories","num_channels"]
    X = cust[feat_cols]
    y = cust["churned"]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                           random_state=42, stratify=y)
    model = XGBClassifier(
        n_estimators=300, learning_rate=0.05,
        max_depth=5, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(y==0).sum()/(y==1).sum(),
        use_label_encoder=False, eval_metric="logloss",
        random_state=42
    )
    model.fit(Xtr, ytr,
              eval_set=[(Xte, yte)],
              verbose=False)

    proba = model.predict_proba(Xte)[:, 1]
    preds = (proba >= 0.5).astype(int)
    auc   = roc_auc_score(yte, proba)
    print(f"  AUC-ROC: {auc:.4f}")
    print(classification_report(yte, preds, target_names=["Active","Churned"]))

    # feature importance plot
    imp = pd.Series(model.feature_importances_, index=feat_cols).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(imp.index, imp.values, color="#3A86FF", edgecolor="none")
    ax.set_title("Churn Model — Feature Importance", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    save_fig(fig, "m2_churn_feature_importance")

    # add churn probability to customer table
    cust.loc[Xte.index, "churn_prob"] = proba
    cust.to_csv(os.path.join(MODEL_DIR, "customer_churn_scores.csv"), index=False)
    save_pkl(model, "model2_churn_xgb")


# ══════════════════════════════════════════════════════════════
# MODEL 3 — RETURNS RISK SCORING (Random Forest)
# ══════════════════════════════════════════════════════════════
def model3_returns_risk(df: pd.DataFrame) -> None:
    print("\n── Model 3: Returns Risk Scoring ───────────────────")
    # simulate return flag: negative profit lines with high discount
    rng = np.random.default_rng(42)
    df2 = df.copy()
    # heuristic return probability
    return_prob = (
        0.3 * (df2["discount"] > 0.3).astype(float)
        + 0.2 * (df2["profit_margin"] < -0.1).astype(float)
        + 0.1 * rng.random(len(df2))
    ).clip(0, 1)
    df2["returned"] = (return_prob > 0.35).astype(int)
    print(f"  Simulated return rate: {df2['returned'].mean():.2%}")

    # encode categoricals
    le_cat = LabelEncoder(); le_sub = LabelEncoder()
    le_ch  = LabelEncoder(); le_seg = LabelEncoder()
    df2["cat_enc"] = le_cat.fit_transform(df2["category"])
    df2["sub_enc"] = le_sub.fit_transform(df2["sub_category"])
    df2["ch_enc"]  = le_ch.fit_transform(df2["channel"])
    df2["seg_enc"] = le_seg.fit_transform(df2["segment"])

    feat_cols = ["sales","quantity","discount","profit_margin",
                 "ship_days","cat_enc","sub_enc","ch_enc","seg_enc"]
    X = df2[feat_cols].fillna(0)
    y = df2["returned"]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                           random_state=42, stratify=y)
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(Xtr, ytr)
    proba = rf.predict_proba(Xte)[:, 1]
    auc   = roc_auc_score(yte, proba)
    print(f"  AUC-ROC: {auc:.4f}")
    print(classification_report(yte, (proba>=0.5).astype(int),
                                 target_names=["No Return","Return"]))

    # feature importance
    imp = pd.Series(rf.feature_importances_, index=feat_cols).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(imp.index, imp.values, color="#FF3A5C", edgecolor="none")
    ax.set_title("Returns Risk — Feature Importance", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    save_fig(fig, "m3_returns_feature_importance")
    save_pkl(rf, "model3_returns_rf")


# ══════════════════════════════════════════════════════════════
# MODEL 4 — PRODUCT RECOMMENDER (Collaborative Filtering)
# ══════════════════════════════════════════════════════════════
def model4_recommender(df: pd.DataFrame) -> None:
    print("\n── Model 4: Product Recommender ────────────────────")
    # build customer × sub_category purchase matrix
    matrix = (df.groupby(["customer_id","sub_category"])["sales"]
                .sum().unstack(fill_value=0))

    print(f"  Matrix shape: {matrix.shape} (customers × sub-categories)")

    # cosine similarity between customers
    from sklearn.metrics.pairwise import cosine_similarity
    cust_sim = cosine_similarity(matrix.values)
    cust_sim_df = pd.DataFrame(cust_sim,
                               index=matrix.index,
                               columns=matrix.index)

    def recommend_for_customer(cid: str, top_n: int = 5) -> list:
        if cid not in cust_sim_df.index:
            return []
        sim_scores = cust_sim_df[cid].sort_values(ascending=False)[1:11]
        similar_customers = sim_scores.index.tolist()
        # aggregate what similar customers bought
        weighted = (matrix.loc[similar_customers]
                    .multiply(sim_scores.values, axis=0)
                    .sum(axis=0))
        # exclude already purchased
        already_bought = matrix.loc[cid][matrix.loc[cid] > 0].index.tolist()
        recs = weighted.drop(already_bought, errors="ignore")
        return recs.sort_values(ascending=False).head(top_n).index.tolist()

    # demo: first 5 customers
    for cid in matrix.index[:5]:
        recs = recommend_for_customer(cid)
        print(f"  {cid[:12]}… → {recs}")

    # visualize sub-category co-purchase heatmap
    sub_corr = matrix.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sub_corr.values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(sub_corr.columns)))
    ax.set_yticks(range(len(sub_corr.index)))
    ax.set_xticklabels(sub_corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(sub_corr.index, fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title("Sub-Category Co-Purchase Similarity", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "m4_copurchase_heatmap")

    save_pkl((matrix, cust_sim_df), "model4_recommender")


# ══════════════════════════════════════════════════════════════
# MODEL 5 — PRICE ELASTICITY
# ══════════════════════════════════════════════════════════════
def model5_price_elasticity(df: pd.DataFrame) -> None:
    print("\n── Model 5: Price Elasticity ───────────────────────")
    # price per unit proxy
    df2 = df.copy()
    df2["unit_price"] = df2["sales"] / df2["quantity"].clip(lower=1)

    agg = (df2.groupby("sub_category")
              .agg(avg_unit_price=("unit_price","mean"),
                   total_qty=("quantity","sum"),
                   avg_discount=("discount","mean"))
              .reset_index())

    # log-log regression: log(qty) ~ log(price)
    agg["log_price"] = np.log(agg["avg_unit_price"].clip(lower=0.01))
    agg["log_qty"]   = np.log(agg["total_qty"].clip(lower=1))

    X = agg[["log_price"]].values
    y = agg["log_qty"].values

    reg = LinearRegression().fit(X, y)
    elasticity = reg.coef_[0]
    print(f"  Overall Price Elasticity: {elasticity:.4f}")
    print(f"  (negative = demand falls as price rises)")

    # per-category elasticity
    results = []
    for cat, grp in df2.groupby("sub_category"):
        grp["unit_price"] = grp["sales"] / grp["quantity"].clip(lower=1)
        g = grp.groupby("order_month").agg(
            avg_price=("unit_price","mean"), total_qty=("quantity","sum")).reset_index()
        if len(g) < 4:
            continue
        lp = np.log(g["avg_price"].clip(lower=0.01))
        lq = np.log(g["total_qty"].clip(lower=1))
        if lp.std() > 0.01:
            r = LinearRegression().fit(lp.values.reshape(-1,1), lq.values)
            results.append({"sub_category": cat, "elasticity": r.coef_[0]})

    elas_df = pd.DataFrame(results).sort_values("elasticity")

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = [("#FF3A5C" if v < -1 else "#FFB703" if v < 0 else "#06D6A0")
              for v in elas_df["elasticity"]]
    ax.barh(elas_df["sub_category"], elas_df["elasticity"],
            color=colors, edgecolor="none")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(-1, color="red", linestyle="--", linewidth=0.8, alpha=0.5, label="Elastic threshold")
    ax.set_xlabel("Price Elasticity Coefficient")
    ax.set_title("Price Elasticity by Sub-Category", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "m5_price_elasticity")
    elas_df.to_csv(os.path.join(MODEL_DIR, "price_elasticity.csv"), index=False)
    save_pkl(reg, "model5_elasticity")


# ══════════════════════════════════════════════════════════════
# MODEL 6 — DEMAND FORECASTING / INVENTORY OPTIMIZER (XGBoost)
# ══════════════════════════════════════════════════════════════
def model6_demand_forecast(df: pd.DataFrame) -> None:
    print("\n── Model 6: Demand / Inventory Forecast ────────────")
    # monthly demand by sub-category
    monthly = (df.groupby(["sub_category","order_year","order_month"])["quantity"]
                 .sum().reset_index())
    monthly["period"] = (monthly["order_year"] * 100 + monthly["order_month"])
    monthly.sort_values(["sub_category","period"], inplace=True)

    def make_lags(grp, lags=3):
        for l in range(1, lags+1):
            grp[f"lag{l}"] = grp["quantity"].shift(l)
        grp["rolling_mean3"] = grp["quantity"].shift(1).rolling(3).mean()
        grp["month_num"]     = grp["order_month"]
        return grp.dropna()

    all_data = (monthly.groupby("sub_category", group_keys=False)
                       .apply(make_lags))

    le = LabelEncoder()
    all_data["cat_enc"] = le.fit_transform(all_data["sub_category"])

    feat_cols = ["cat_enc","lag1","lag2","lag3","rolling_mean3","month_num"]
    X = all_data[feat_cols]
    y = all_data["quantity"]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb = XGBRegressor(n_estimators=300, learning_rate=0.05,
                       max_depth=5, subsample=0.8,
                       random_state=42)
    xgb.fit(Xtr, ytr)
    preds = xgb.predict(Xte)
    regression_metrics(yte, preds, "Demand Forecast")

    # reorder point calculation
    top_cats = df["sub_category"].value_counts().head(8).index
    reorder = (df[df["sub_category"].isin(top_cats)]
               .groupby("sub_category")["quantity"]
               .agg(avg_monthly_demand="mean", std_demand="std"))
    reorder["safety_stock"]   = (1.65 * reorder["std_demand"]).round()   # 95% service level
    reorder["reorder_point"]  = (reorder["avg_monthly_demand"] + reorder["safety_stock"]).round()
    print("\n  Reorder Points:")
    print(reorder.to_string())
    reorder.to_csv(os.path.join(MODEL_DIR, "reorder_points.csv"))

    # feature importance
    imp = pd.Series(xgb.feature_importances_, index=feat_cols).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(imp.index, imp.values, color="#7B5EA7", edgecolor="none")
    ax.set_title("Demand Forecast — Feature Importance", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    save_fig(fig, "m6_demand_feature_importance")
    save_pkl(xgb, "model6_demand_xgb")


# ── main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv(CLEAN_PATH, parse_dates=["order_date","ship_date"])

    model1_revenue_forecast(df)
    model2_churn_prediction(df)
    model3_returns_risk(df)
    model4_recommender(df)
    model5_price_elasticity(df)
    model6_demand_forecast(df)

    print(f"\n[✓] All models saved → {MODEL_DIR}")
