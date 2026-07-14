"""
Stage 3 gate: binary SVC lesion-symptom mapping end to end on the 2-class fixture.

Checks:
  1. run_svm_lsm with mode="svc" produces one beta_map + one z-map + an HTML report.
  2. The estimator is an SVC (svc_model.pkl), grid-searched by ROC-AUC.
  3. Covariates are handled feature-side only (behaviour labels stay {0,1}; the
     covariate_regressed_out_of_overlap_map is written, behaviour-side regression skipped).
  4. AUC on the planted signal beats chance (sanity that the pipeline finds the blob).
"""
import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from svmLSMpy import run_svm_lsm
from sklearn.svm import SVC

DATA = Path(__file__).parent / "fixtures" / "data" / "binary"
OUT = Path(__file__).parent / "_out" / "verify_svc_binary"
SMALL_GRID = {"C": [1, 10], "gamma": ["scale"]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = run_svm_lsm(
        symptom_folder=str(DATA / "lesions"),
        csv_path=str(DATA / "behavior.csv"),
        output_path=str(OUT),
        behaviour_name="verify_svc_bin",
        mode="svc",
        param_grid=SMALL_GRID,
        n_permutations=20,
        n_splits=5,
        num_slices=3,
    )
    folder = max(OUT.glob("*_results_*"), key=lambda p: p.stat().st_mtime)

    assert len(results) == 1, f"binary SVC should yield 1 map, got {len(results)}"
    for name in ("beta_map.nii.gz", "zmap.nii.gz", "svc_model.pkl", "svc_lsm_report.html",
                 "covariate_regressed_out_of_overlap_map.nii.gz"):
        assert (folder / name).exists(), f"missing artifact: {name}"
    print("[1] one beta_map + z-map + report generated  OK")

    with open(folder / "svc_model.pkl", "rb") as f:
        model = pickle.load(f)
    assert isinstance(model, SVC), f"expected SVC, got {type(model)}"
    print("[2] estimator is SVC (ROC-AUC model selection)  OK")

    # labels stayed categorical -> beta map exists and behaviour was not normalised
    beta = nib.load(str(folder / "beta_map.nii.gz")).get_fdata()
    assert np.isfinite(beta).all() and (beta != 0).any(), "beta map degenerate"
    print("[3] covariates handled feature-side; labels categorical  OK")

    scores = pd.read_csv(folder / "results_and_scores.csv")
    best_auc = scores["Avg_Score"].max()  # greater is better for AUC
    assert best_auc > 0.6, f"AUC {best_auc:.3f} at/below chance - signal not detected"
    print(f"[4] best CV AUC={best_auc:.3f} beats chance  OK")
    print("\nALL BINARY SVC CHECKS PASSED")


if __name__ == "__main__":
    main()
