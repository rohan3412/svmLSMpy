"""
Stage 4 gate: multiclass one-vs-rest SVC on the 3-class fixture (Option A).

Checks:
  1. run_svm_lsm with mode="svc" routes to the multiclass task and returns K=3 MapResults.
  2. Exactly ONE grid search runs (single results_and_scores.csv, no per-class suffix)
     -> Option A shared-hyperparameter selection.
  3. Each class yields its own beta_map, z-map, SVC model, and HTML report (suffixed).
  4. Multiclass CV balanced accuracy beats chance (1/3).
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from svmLSMpy import run_svm_lsm

DATA = Path(__file__).parent / "fixtures" / "data" / "threeclass"
OUT = Path(__file__).parent / "_out" / "verify_svc_multiclass"
SMALL_GRID = {"C": [1, 10], "gamma": ["scale"]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = run_svm_lsm(
        symptom_folder=str(DATA / "lesions"),
        csv_path=str(DATA / "behavior.csv"),
        output_path=str(OUT),
        behaviour_name="verify_svc_3c",
        mode="svc",
        param_grid=SMALL_GRID,
        n_permutations=20,
        n_splits=5,
        num_slices=3,
    )
    folder = max(OUT.glob("*_results_*"), key=lambda p: p.stat().st_mtime)

    assert len(results) == 3, f"expected 3 one-vs-rest maps, got {len(results)}"
    print("[1] routed to multiclass; 3 one-vs-rest maps  OK")

    grid_csvs = list(folder.glob("results_and_scores*.csv"))
    assert grid_csvs == [folder / "results_and_scores.csv"], \
        f"Option A expects ONE grid search, found {[p.name for p in grid_csvs]}"
    print("[2] single shared grid search (Option A)  OK")

    for c in (0, 1, 2):
        for name in (f"beta_map_class{c}.nii.gz", f"zmap_class{c}.nii.gz",
                     f"svc_model_class{c}.pkl", f"svc_lsm_report_class{c}.html"):
            assert (folder / name).exists(), f"missing per-class artifact: {name}"
    print("[3] per-class beta/z-map/model/report written  OK")

    scores = pd.read_csv(folder / "results_and_scores.csv")
    best_bal_acc = scores["Avg_Score"].max()
    assert best_bal_acc > 0.5, f"balanced accuracy {best_bal_acc:.3f} near chance (0.33)"
    print(f"[4] best CV balanced accuracy={best_bal_acc:.3f} beats chance  OK")
    print("\nALL MULTICLASS SVC CHECKS PASSED")


if __name__ == "__main__":
    main()
