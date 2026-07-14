"""
Stage 0 golden baseline: run the full SVR pipeline on the synthetic fixture and
capture the DETERMINISTIC artifacts (grid-search best params + beta/coef map).

The final z-map is intentionally NOT used as the gate: the permutation loop uses
shuffle(random_state=None), so the null distribution - and thus the z-map - varies
run to run. best_params and coef_map (beta_map.nii.gz) are fully deterministic
given the fixed KFold(random_state=42), so those are what the Stage 1 refactor
must preserve exactly.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from svmLSMpy import run_svm_lsm

DATA = Path(__file__).parent / "fixtures" / "data" / "svr"
OUT = Path(__file__).parent / "_out" / "svr"
GOLDEN = Path(__file__).parent / "fixtures" / "golden"

SMALL_GRID = {"C": [1, 10], "gamma": ["scale"], "epsilon": [0.1]}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        run_svm_lsm(
            symptom_folder=str(DATA / "lesions"),
            csv_path=str(DATA / "behavior.csv"),
            max_score=100,
            output_path=str(OUT),
            behaviour_name="fixture_svr",
            mode="svr",
            param_grid=SMALL_GRID,
            n_permutations=30,
            n_splits=5,
            num_slices=3,
            # Disable lesion-side covariate regression to keep the golden baseline
            # on a known-working path (the overlap map write path was previously buggy).
            regress_out_covariates_on_lesions=False,
        )
    except TypeError as e:
        # Known env blocker: atlasreader calls nilearn check_niimg(atleast_4d=...),
        # removed in the installed nilearn. This is the atlas/report tail, which runs
        # AFTER the deterministic core artifacts (beta_map, results csv) are written.
        if "atleast_4d" in str(e):
            print(f"\n[baseline] tolerated downstream atlas/report failure: {e}")
        else:
            raise
    # newest results folder
    folder = max(OUT.glob("*_results_*"), key=lambda p: p.stat().st_mtime)
    return folder


def capture(folder):
    GOLDEN.mkdir(parents=True, exist_ok=True)
    beta = nib.load(str(folder / "beta_map.nii.gz")).get_fdata()
    np.save(GOLDEN / "svr_beta_map.npy", beta)

    scores = pd.read_csv(folder / "results_and_scores.csv")
    best = scores.loc[scores["Avg_Score"].idxmin()]
    best_params = {"C": best["C"], "Gamma": best["Gamma"], "Epsilon": best["Epsilon"]}
    (GOLDEN / "svr_best_params.txt").write_text(str(best_params))
    print("\n=== GOLDEN BASELINE CAPTURED ===")
    print("results folder:", folder.name)
    print("best params:", best_params)
    print("beta_map: shape", beta.shape, "sum", float(beta.sum()), "nonzero", int((beta != 0).sum()))


if __name__ == "__main__":
    capture(run())
