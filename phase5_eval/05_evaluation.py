"""
=============================================================
PHASE 5 — MODEL EVALUATION & VALIDATION
=============================================================
• Cross-validation for all models
• SHAP explainability (XGBoost models)
• ROC / PR curves for classifiers
• Residual analysis for regressors
• Business KPI lift table
=============================================================
"""

import pandas as pd
import numpy as np
import pickle, os, warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection   import StratifiedKFold, KFold, cross_validate
from sklearn.metrics           import (
    roc_curve, auc, precision_recall_curve,
    average_precision_score, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.preprocessing     import LabelEncoder
from sklearn.ensemble          import RandomForestClassifier
from xgboost                   import XGBClassifier, XGBRegressor

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[!] shap not installed — SHAP plots will be skipped")

# ── paths ────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT_DIR, "data")
CLEAN_PATH = os.path.join(DATA_DIR, "superstore_clean.csv")
MODEL_DIR  = os.path.join(ROOT_DIR, "models")
EVAL_DIR   = os.path.join(ROOT_DIR, "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

def save_fig(fig, name):
    fig.savefig(os.path.join(EVAL_DIR, f"{name}.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [✓] {name}.png")

def load_pkl(name):
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


# ══════════════════════════════════════════════════════════════
# BUILD FEATURE SETS (mirrors Phase 4 logic)
# ══════════════════════════════════════════════════════════════
def build_churn_features(df):
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
    cust["churned"] = (cust["recency"] > 180).astype(int)
    feat_cols = ["recency","frequency","monetary","total_profit",
                 "avg_discount","avg_margin","num_categories","num_channels"]
    return cust[feat_cols], cust["churned"], feat_cols


def build_return_features(df):
    rng = np.random.default_rng(42)
    df2 = df.copy()
    rp = (0.3*(df2["discount"]>0.3).astype(float)
          + 0.2*(df2["profit_margin"]<-0.1).astype(float)
          + 0.1*rng.random(len(df2))).clip(0,1)
    df2["returned"] = (rp > 0.35).astype(int)
    for col, le in [("category", LabelEncoder()), ("sub_category", LabelEncoder()),
                    ("channel",  LabelEncoder()), ("segment", LabelEncoder())]:
        df2[col+"_enc"] = le.fit_transform(df2[col])
    feat_cols = ["sales","quantity","discount","profit_margin",
                 "ship_days","category_enc","sub_category_enc",
                 "channel_enc","segment_enc"]
    return df2[feat_cols].fillna(0), df2["returned"], feat_cols


# ══════════════════════════════════════════════════════════════
# 1. CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════
def cross_validate_models(df):
    print("\n── 1. Cross-Validation ─────────────────────────────")

    # Churn model
    X_churn, y_churn, _ = build_churn_features(df)
    cv_churn = cross_validate(
        XGBClassifier(n_estimators=200, learning_rate=0.05,
                      use_label_encoder=False, eval_metric="logloss",
                      random_state=42),
        X_churn, y_churn,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring=["roc_auc","f1","precision","recall"],
        return_train_score=True
    )
    print("\n  [Churn Model — 5-Fold Stratified CV]")
    for metric in ["test_roc_auc","test_f1","test_precision","test_recall"]:
        vals = cv_churn[metric]
        print(f"    {metric:20s}: {vals.mean():.4f} ± {vals.std():.4f}")

    # Returns model
    X_ret, y_ret, _ = build_return_features(df)
    cv_ret = cross_validate(
        RandomForestClassifier(n_estimators=100, class_weight="balanced",
                               random_state=42, n_jobs=-1),
        X_ret, y_ret,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring=["roc_auc","f1"],
        return_train_score=True
    )
    print("\n  [Returns Risk Model — 5-Fold Stratified CV]")
    for metric in ["test_roc_auc","test_f1"]:
        vals = cv_ret[metric]
        print(f"    {metric:20s}: {vals.mean():.4f} ± {vals.std():.4f}")

    # Plot CV scores
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, scores, title in zip(
        axes,
        [cv_churn["test_roc_auc"], cv_ret["test_roc_auc"]],
        ["Churn Model — AUC per Fold", "Returns Model — AUC per Fold"]
    ):
        folds = range(1, len(scores)+1)
        ax.bar(folds, scores, color="#3A86FF", edgecolor="none", alpha=0.8)
        ax.axhline(scores.mean(), color="red", linestyle="--",
                   linewidth=1.5, label=f"Mean={scores.mean():.3f}")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Fold")
        ax.set_ylabel("AUC-ROC")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "01_cross_validation_auc")


# ══════════════════════════════════════════════════════════════
# 2. ROC & PRECISION-RECALL CURVES
# ══════════════════════════════════════════════════════════════
def roc_pr_curves(df):
    print("\n── 2. ROC & PR Curves ──────────────────────────────")
    from sklearn.model_selection import train_test_split

    results = {}
    configs = [
        ("Churn",   build_churn_features,
         XGBClassifier(n_estimators=200, use_label_encoder=False,
                       eval_metric="logloss", random_state=42)),
        ("Returns", build_return_features,
         RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                random_state=42, n_jobs=-1)),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    colors = ["#3A86FF", "#FF3A5C"]

    for idx, (label, feat_fn, clf) in enumerate(configs):
        X, y, _ = feat_fn(df)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                               random_state=42, stratify=y)
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)[:, 1]

        # ROC
        fpr, tpr, _ = roc_curve(yte, proba)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color=colors[idx], linewidth=2,
                     label=f"{label} (AUC={roc_auc:.3f})")

        # PR
        prec, rec, _ = precision_recall_curve(yte, proba)
        ap = average_precision_score(yte, proba)
        axes[1].plot(rec, prec, color=colors[idx], linewidth=2,
                     label=f"{label} (AP={ap:.3f})")

        results[label] = {"auc": roc_auc, "ap": ap}

    axes[0].plot([0,1],[0,1],"k--",linewidth=0.8)
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curves", fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curves", fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    save_fig(fig, "02_roc_pr_curves")
    return results


# ══════════════════════════════════════════════════════════════
# 3. SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════
def shap_explainability(df):
    if not SHAP_AVAILABLE:
        print("\n── 3. SHAP [SKIPPED — install shap] ────────────────")
        return

    print("\n── 3. SHAP Explainability ──────────────────────────")
    from sklearn.model_selection import train_test_split

    X, y, feat_cols = build_churn_features(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                           random_state=42, stratify=y)
    model = XGBClassifier(n_estimators=200, use_label_encoder=False,
                          eval_metric="logloss", random_state=42)
    model.fit(Xtr, ytr)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xte)

    # Summary plot
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, Xte, feature_names=feat_cols,
                      plot_type="bar", show=False)
    plt.title("SHAP Feature Importance — Churn Model",
              fontsize=11, fontweight="bold")
    plt.tight_layout()
    save_fig(plt.gcf(), "03_shap_summary_bar")

    # Beeswarm
    shap.summary_plot(shap_values, Xte, feature_names=feat_cols, show=False)
    plt.title("SHAP Beeswarm — Churn Model", fontsize=11, fontweight="bold")
    plt.tight_layout()
    save_fig(plt.gcf(), "03_shap_beeswarm")

    # Single prediction waterfall
    idx = 0
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[idx],
            base_values=explainer.expected_value,
            data=Xte.iloc[idx].values,
            feature_names=feat_cols
        ), show=False
    )
    plt.title("SHAP Waterfall — Single Prediction", fontsize=10, fontweight="bold")
    plt.tight_layout()
    save_fig(plt.gcf(), "03_shap_waterfall")


# ══════════════════════════════════════════════════════════════
# 4. BUSINESS KPI LIFT TABLE
# ══════════════════════════════════════════════════════════════
def business_kpi_lift(df):
    print("\n── 4. Business KPI Lift Table ──────────────────────")
    from sklearn.model_selection import train_test_split

    X, y, feat_cols = build_churn_features(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                           random_state=42, stratify=y)
    model = XGBClassifier(n_estimators=200, use_label_encoder=False,
                          eval_metric="logloss", random_state=42)
    model.fit(Xtr, ytr)

    # join churn prob back to customer data
    ref_date = df["order_date"].max()
    cust = (df.groupby("customer_id")
              .agg(total_profit=("profit","sum"),
                   total_sales=("sales","sum"))
              .reset_index())
    Xtest_df = Xte.copy()
    Xtest_df["churn_prob"] = model.predict_proba(Xte)[:, 1]
    Xtest_df["actual_churned"] = yte.values

    # Decile lift table
    Xtest_df["decile"] = pd.qcut(Xtest_df["churn_prob"], 10,
                                  labels=False, duplicates="drop")
    lift = (Xtest_df.groupby("decile")
                    .agg(n=("actual_churned","count"),
                         churned=("actual_churned","sum"),
                         avg_churn_prob=("churn_prob","mean"))
                    .reset_index())
    lift["churn_rate"]  = lift["churned"] / lift["n"]
    lift["cum_churned"] = lift["churned"][::-1].cumsum()[::-1]
    lift["lift"]        = lift["churn_rate"] / yte.mean()
    lift.sort_values("decile", ascending=False, inplace=True)

    print("\n  Decile Lift Table (top 10 most at-risk deciles):")
    print(lift[["decile","n","churned","churn_rate","lift","avg_churn_prob"]].to_string(index=False))

    # Plot lift curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(lift["decile"].astype(str), lift["lift"],
           color="#3A86FF", edgecolor="none", alpha=0.8)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="Baseline lift=1")
    ax.set_xlabel("Decile (10=highest churn risk)")
    ax.set_ylabel("Lift")
    ax.set_title("Churn Model — Decile Lift Chart", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    save_fig(fig, "04_decile_lift_chart")

    lift.to_csv(os.path.join(EVAL_DIR, "churn_lift_table.csv"), index=False)


# ══════════════════════════════════════════════════════════════
# 5. FORECAST RESIDUAL ANALYSIS
# ══════════════════════════════════════════════════════════════
def residual_analysis(df):
    print("\n── 5. Forecast Residual Analysis ───────────────────")
    from sklearn.model_selection import train_test_split
    monthly = (df.groupby(["order_year","order_month"])["sales"]
                 .sum().reset_index())
    monthly["lag1"] = monthly["sales"].shift(1)
    monthly["lag3"] = monthly["sales"].shift(3)
    monthly["month_num"] = monthly["order_month"]
    monthly.dropna(inplace=True)

    X = monthly[["lag1","lag3","month_num"]]
    y = monthly["sales"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=200, random_state=42)
    model.fit(Xtr, ytr)
    preds    = model.predict(Xte)
    residuals = yte.values - preds

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Residuals vs predicted
    axes[0].scatter(preds, residuals, alpha=0.6, color="#3A86FF", s=30, edgecolors="none")
    axes[0].axhline(0, color="red", linewidth=1)
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Fitted", fontsize=10, fontweight="bold")
    axes[0].grid(alpha=0.3)

    # Histogram of residuals
    axes[1].hist(residuals, bins=20, color="#06D6A0", edgecolor="none", alpha=0.8)
    axes[1].set_xlabel("Residual")
    axes[1].set_title("Residual Distribution", fontsize=10, fontweight="bold")
    axes[1].grid(alpha=0.3)

    # Actual vs predicted
    axes[2].scatter(yte, preds, alpha=0.6, color="#FFB703", s=30, edgecolors="none")
    mn = min(yte.min(), preds.min()); mx = max(yte.max(), preds.max())
    axes[2].plot([mn,mx],[mn,mx],"r--",linewidth=1)
    axes[2].set_xlabel("Actual"); axes[2].set_ylabel("Predicted")
    axes[2].set_title("Actual vs Predicted", fontsize=10, fontweight="bold")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    save_fig(fig, "05_residual_analysis")

    mae  = mean_absolute_error(yte, preds)
    rmse = np.sqrt(mean_squared_error(yte, preds))
    r2   = r2_score(yte, preds)
    mape = np.mean(np.abs(residuals / yte.clip(lower=1))) * 100
    print(f"  MAE={mae:.1f}  RMSE={rmse:.1f}  R²={r2:.4f}  MAPE={mape:.2f}%")


# ── main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv(CLEAN_PATH, parse_dates=["order_date","ship_date"])

    cross_validate_models(df)
    roc_pr_curves(df)
    shap_explainability(df)
    business_kpi_lift(df)
    residual_analysis(df)

    print(f"\n[✓] Evaluation outputs saved → {EVAL_DIR}")
