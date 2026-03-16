# End-to-End Profit Optimization System
### Multi-Channel Retailer — Superstore Dataset

---

## Project Structure

```
profit_optimizer/
│
├── run_all.py                          ← Master pipeline runner
├── requirements.txt
│
├── data/                               ← Put your CSV here
│   └── Sample_-_Superstore.csv
│
├── phase1_data/
│   └── 01_ingest_and_clean.py          ← Load, clean, validate, derive features
│
├── phase2_sql/
│   └── 02_sql_warehouse.py             ← Build star schema + 10 analytical SQL queries
│
├── phase3_stats/
│   └── 03_statistical_analysis.py      ← Correlation, ANOVA, A/B test, RFM clustering
│
├── phase4_ml/
│   └── 04_ml_models.py                 ← 6 ML/DL models (churn, revenue, returns, etc.)
│
├── phase5_eval/
│   └── 05_evaluation.py                ← CV, ROC/PR, SHAP, lift table
│
├── phase6_powerbi/
│   └── 06_powerbi_exports.py           ← CSV exports + all DAX measures
│
└── phase7_recommendations/
    └── 07_recommendations.py           ← Marketing, pricing, inventory, channel actions
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place the CSV
cp "Sample - Superstore.csv" data/

# 3. Run all phases
cd profit_optimizer
python run_all.py

# Or run a single phase
python phase1_data/01_ingest_and_clean.py
```

---

## Phase-by-Phase Summary

| Phase | Script | What it does | Key outputs |
|-------|--------|-------------|-------------|
| 1 | `01_ingest_and_clean.py` | Load CSV, parse dates, derive metrics, add channel simulation | `superstore_clean.csv` |
| 2 | `02_sql_warehouse.py` | Build SQLite star schema, run 10 profit SQL queries | `superstore_dw.db` |
| 3 | `03_statistical_analysis.py` | Correlation heatmap, ANOVA, A/B discount test, RFM k-means | 6 charts + `rfm_segments.csv` |
| 4 | `04_ml_models.py` | Revenue forecast, churn, returns risk, recommender, price elasticity, demand | 6 `.pkl` models |
| 5 | `05_evaluation.py` | 5-fold CV, ROC/PR curves, SHAP explainability, decile lift | Charts + `churn_lift_table.csv` |
| 6 | `06_powerbi_exports.py` | 5 Power BI CSVs + full DAX measure library | `powerbi_exports/` folder |
| 7 | `07_recommendations.py` | Actionable decisions per customer/product/channel + HTML report | `executive_summary.html` |

---

## Models Built

| Model | Algorithm | Target | Business Use |
|-------|-----------|--------|-------------|
| Revenue Forecast | Prophet / XGBoost | Monthly sales | Budget planning |
| Churn Prediction | XGBoost Classifier | Will customer churn? | Win-back campaigns |
| Returns Risk | Random Forest | Will this order be returned? | Fulfilment prioritization |
| Product Recommender | Collaborative Filtering | What to cross-sell? | Personalization engine |
| Price Elasticity | Log-Log Regression | Price sensitivity | Dynamic pricing |
| Demand Forecast | XGBoost Regressor | Monthly qty by SKU | Inventory reorder |


- **Impact**: Identified $X recoverable profit from loss-making lines + high-discount orders
