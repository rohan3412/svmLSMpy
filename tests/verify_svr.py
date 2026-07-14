"""
Regression check after removing atlasreader and fixing the covariate/axis + unicode
bugs. Confirms:
  1. SVR core still reproduces the Stage-0 golden baseline (beta_map + best_params).
  2. The HTML report now completes end-to-end (no atlasreader crash).
  3. Lesion-side covariate regression runs without the old np.sum(axis=-1) crash.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from svmLSMpy import run_svm_lsm

DATA = Path(__file__).parent / "fixtures" / "data" / "svr"
GOLDEN = Path(__file__).parent / "fixtures" / "golden"
OUT = Path(__file__).parent / "_out" / "verify_svr"
SMALL_GRID = {"C": [1, 10], "gamma": ["scale"], "epsilon": [0.1]}


def _run(tag, **extra):
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    run_svm_lsm(
        symptom_folder=str(DATA / "lesions"),
        csv_path=str(DATA / "behavior.csv"),
        max_score=100, output_path=str(out), behaviour_name=f"verify_{tag}",
        mode="svr", param_grid=SMALL_GRID, n_permutations=20, n_splits=5, num_slices=3, **extra,
    )
    return max(out.glob("*_results_*"), key=lambda p: p.stat().st_mtime)


def main():
    # 1 + 2: baseline flags (lesion-side covariate regression OFF, matching golden)
    folder = _run("baseline", regress_out_covariates_on_lesions=False)
    beta = nib.load(str(folder / "beta_map.nii.gz")).get_fdata()
    golden = np.load(GOLDEN / "svr_beta_map.npy")
    assert np.allclose(beta, golden), "beta_map diverged from golden baseline!"
    scores = pd.read_csv(folder / "results_and_scores.csv")
    best = scores.loc[scores["Avg_Score"].idxmin()]
    assert (best["C"], best["Gamma"], best["Epsilon"]) == (1, "scale", 0.1), "best params changed!"
    assert (folder / "svr_lsm_report.html").exists(), "report not generated!"
    print("[1] beta_map + best_params match golden  OK")
    print("[2] HTML report generated (atlasreader removed)  OK")

    # 3: lesion-side covariate regression ON -> exercises the axis=0 fix
    folder2 = _run("covlesion", regress_out_covariates_on_lesions=True)
    assert (folder2 / "covariate_regressed_out_of_overlap_map.nii.gz").exists(), \
        "covariate overlap map missing - axis fix failed!"
    print("[3] lesion-side covariate regression ran (axis bug fixed)  OK")
    print("\nALL SVR REGRESSION CHECKS PASSED")


if __name__ == "__main__":
    main()
