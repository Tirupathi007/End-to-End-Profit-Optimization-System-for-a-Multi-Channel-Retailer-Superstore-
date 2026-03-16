"""
=============================================================
RUN_ALL.PY — MASTER PIPELINE
=============================================================
Runs all 7 phases in order.
Usage:
    cd profit_optimizer
    python run_all.py

Requirements:
    pip install pandas numpy matplotlib scipy scikit-learn
               xgboost shap prophet (optional)
=============================================================
"""

import subprocess, sys, time, os

PHASES = [
    ("Phase 1 — Ingest & Clean",          "phase1_data/01_ingest_and_clean.py"),
    ("Phase 2 — SQL Warehouse",           "phase2_sql/02_sql_warehouse.py"),
    ("Phase 3 — Statistical Analysis",   "phase3_stats/03_statistical_analysis.py"),
    ("Phase 4 — ML / DL Models",         "phase4_ml/04_ml_models.py"),
    ("Phase 5 — Evaluation & SHAP",      "phase5_eval/05_evaluation.py"),
    ("Phase 6 — Power BI Exports",       "phase6_powerbi/06_powerbi_exports.py"),
    ("Phase 7 — Recommendations",        "phase7_recommendations/07_recommendations.py"),
]

def run_phase(name, script):
    print(f"\n{'='*60}")
    print(f"  ▶  {name}")
    print(f"{'='*60}")
    start = time.time()
    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,
        text=True
    )
    elapsed = time.time() - start
    status = "✓ Done" if result.returncode == 0 else "✗ FAILED"
    print(f"\n  [{status}]  {elapsed:.1f}s")
    return result.returncode == 0

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║     End-to-End Profit Optimization System                   ║
║     Superstore Multi-Channel Retailer                       ║
╚══════════════════════════════════════════════════════════════╝
""")
    results = []
    for name, script in PHASES:
        success = run_phase(name, script)
        results.append((name, success))
        if not success:
            print(f"\n[!] Pipeline halted at: {name}")
            print("    Fix the error above and re-run from this phase.")
            break

    print(f"\n{'='*60}")
    print("  PIPELINE SUMMARY")
    print(f"{'='*60}")
    for name, ok in results:
        icon = "✓" if ok else "✗"
        print(f"  {icon}  {name}")

    all_ok = all(ok for _, ok in results)
    print(f"\n{'  [ALL PHASES COMPLETE]' if all_ok else '  [PIPELINE INCOMPLETE]'}")
    print("""
  Output directories:
    data/superstore_clean.csv
    data/superstore_dw.db
    data/stats_outputs/
    data/models/
    data/evaluation/
    data/powerbi_exports/
    data/recommendations/executive_summary.html
""")
