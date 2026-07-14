from .svm_lsm import svm_lsm
from .task import SVR_TASK


def svr_lsm(features, behaviors, masker, output_folder, param_grid, n_permutations=1, alpha=0.05, n_splits=5):
    """
    Perform SVR-based lesion-symptom mapping with K-fold cross-validation and
    permutation testing.

    Thin wrapper over the generic svm_lsm engine, pinned to SVR_TASK. Kept so existing
    callers and imports (from .svr_lsm import svr_lsm) keep working unchanged.
    """
    # svm_lsm is list-first (one MapResult per decision map); SVR yields exactly one.
    results = svm_lsm(
        features, behaviors, masker, output_folder, param_grid, SVR_TASK,
        n_permutations=n_permutations, alpha=alpha, n_splits=n_splits,
    )
    r = results[0]
    return r.best_params, r.coef_map, r.nifti_zmap, r.zmap
