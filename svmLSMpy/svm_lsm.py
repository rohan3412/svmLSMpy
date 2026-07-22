from dataclasses import dataclass
from itertools import product
import pickle
import time
from pathlib import Path

import pandas as pd
import numpy as np
from nilearn import masking
import nibabel as nib
from nilearn.image import threshold_img
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.utils import shuffle
from tqdm import tqdm
from joblib import Parallel, delayed

import warnings
warnings.filterwarnings("ignore")

from .util import easy_time, easy_eta
from .plot_tuning import plot_tuning


@dataclass
class MapResult:
    """One support-vector -> z-map result. A single-decision run (SVR / binary SVC)
    yields a list of one; one-vs-rest multiclass SVC yields one per class."""
    label: str            # "" for single map; class name for one-vs-rest
    suffix: str           # filename suffix used for this map's artifacts ("" for single)
    best_params: dict
    coef_map: np.ndarray
    nifti_zmap: object    # nibabel image
    zmap: np.ndarray
    no_signal: bool = False  # True when the map degenerated to an all-zero z-map (no signal)


def _cv_score(features, behaviors, task, params, n_splits, verbose=False):
    """Evaluate one hyper-parameter point by K-fold CV using the task's scorer.
    Returns (avg_score, per_fold_scores, per_fold_support_vector_counts, avg_sv_fraction).
    avg_sv_fraction is the mean over folds of len(support_)/len(training set) - a diagnostic
    only (1.0 = every training sample is a support vector); it is NOT a selection criterion."""
    scores = []
    no_of_sv = []
    sv_fracs = []

    cv = task.make_cv(n_splits)
    splits = cv.split(features, behaviors) if task.stratified else cv.split(features)

    for fold_idx, (train_idx, test_idx) in enumerate(splits, 1):
        if verbose:
            print(f"Split :{fold_idx}/{n_splits}")
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = behaviors[train_idx], behaviors[test_idx]

        est = task.make_estimator(params)
        est.fit(X_train, y_train)

        if verbose:
            print(f"\tno. of support vectors : {len(est.support_)}/{len(X_train)}", )
        no_of_sv.append(len(est.support_))
        sv_fracs.append(len(est.support_) / len(X_train))

        y_pred = est.decision_function(X_test) if task.score_uses_decision else est.predict(X_test)
        score = task.scorer(y_test, y_pred)

        if verbose:
            print(f"\tscore : {score}", )
        scores.append(score)

    return float(np.mean(scores)), scores, no_of_sv, float(np.mean(sv_fracs))


def _eval_one_combo(combo, param_names, features, behaviors, task, n_splits):
    """Evaluate one hyperparameter combination. Returns all info needed for logging."""
    params = dict(zip(param_names, combo))
    avg_score, scores, no_of_sv, avg_sv_frac = _cv_score(
        features, behaviors, task, params, n_splits)
    avg_no_of_sv = float(np.mean(no_of_sv))
    return params, combo, avg_score, scores, no_of_sv, avg_no_of_sv, avg_sv_frac


def _grid_search(features, behaviors, output_folder, param_grid, task, n_splits, suffix="", n_jobs=1):
    """Grid-search hyper-parameters by K-fold CV using the task's scorer. Writes
    results_and_scores{suffix}.csv and returns (best_params, evaluations) - the latter a
    list of {**params, 'avg_score', 'source': 'grid'} dicts for the tuning plot."""
    print(f"Running {task.name} analysis...")

    # Perform grid search with K-fold cross-validation
    print("Performing grid search with K-fold cross-validation...")

    param_names = list(task.param_names)
    param_combinations = list(product(*[param_grid[k] for k in param_names]))

    best_params = dict(zip(param_names, param_combinations[0]))
    best_score = task.worst_score()
    best_iteration = 1
    best_sv_frac = 1.0

    all_scores = []
    evaluations = []
    num_iter = len(param_combinations)

    import os
    actual_jobs = os.cpu_count() if n_jobs == -1 else n_jobs
    print(f"Evaluating {num_iter} parameter combinations with {actual_jobs} jobs...")
    grid_time = time.time()

    raw_results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_eval_one_combo)(combo, param_names, features, behaviors, task, n_splits)
        for combo in tqdm(param_combinations, desc="Queuing grid search")
    )

    # Process results sequentially to find best params
    for idx, (params, combo, avg_score, scores, no_of_sv, avg_no_of_sv, avg_sv_frac) in enumerate(raw_results, 1):
        all_scores.append((idx, *combo, avg_score, scores, avg_no_of_sv, no_of_sv, round(avg_sv_frac, 4)))
        evaluations.append({**params, "avg_score": avg_score, "source": "grid"})

        if task.is_better(avg_score, best_score):
            best_iteration = idx
            best_score = avg_score
            best_params = params
            best_sv_frac = avg_sv_frac

    elapsed = time.time() - grid_time
    print(f"\nGrid search completed in {easy_time(int(elapsed))}.")

    columns = ["Iteration"] + [n.capitalize() for n in param_names] + [
        "Avg_Score", "Scores", "Avg_Support_Vectors", "Support_Vectors", "SV_Fraction"
    ]

    df = pd.DataFrame(all_scores, columns=columns)

    # Save to CSV
    output_path = output_folder / f'results_and_scores{suffix}.csv'
    df.to_csv(output_path, index=False)
    print(f"Results and scores saved to {output_path}")

    print(f"\nBest parameters found: {best_params} with score {best_score:.4f} in iteration {best_iteration}/{num_iter}")
    print(f"Selected model's mean SV fraction: {best_sv_frac:.0%} of the training set "
          f"(diagnostic only; selection is by CV score. {'100% is expected under strong regularization / small C.' if best_sv_frac >= 0.999 else ''})")
    return best_params, evaluations


def _resolve_numeric_gamma(gamma, features):
    """Map sklearn's string gammas to their numeric value so they can seed the simplex."""
    if gamma == "scale":
        return 1.0 / (features.shape[1] * features.var())
    if gamma == "auto":
        return 1.0 / features.shape[1]
    return float(gamma)


def _nelder_mead_refine(features, behaviors, task, best_params, grid_best_score,
                        n_splits, evaluations):
    """Locally refine the best grid point with a derivative-free Nelder-Mead simplex -
    the sound (gradient-free) stand-in for the BFGS the objective can't support.

    Optimizes [log10(C), log10(gamma)] for SVC and [log10(C), log10(gamma), epsilon] for
    SVR. Every evaluated point is appended to `evaluations` (source='refine') for the plot.
    Returns the better of the grid best and the simplex best (guarded against CV noise)."""
    refine_epsilon = "epsilon" in task.param_names
    sign = -1.0 if task.greater_is_better else 1.0
    LOG_LO, LOG_HI, EPS_MIN, PENALTY = -4.0, 3.0, 1e-3, 1e6

    x0 = [np.log10(float(best_params["C"])),
          np.log10(_resolve_numeric_gamma(best_params["gamma"], features))]
    if refine_epsilon:
        x0.append(float(best_params.get("epsilon", 0.1)))

    def objective(x):
        logC, logG = x[0], x[1]
        if not (LOG_LO <= logC <= LOG_HI and LOG_LO <= logG <= LOG_HI):
            return PENALTY
        params = {"C": float(10 ** logC), "gamma": float(10 ** logG)}
        if refine_epsilon:
            if x[2] < EPS_MIN:
                return PENALTY
            params["epsilon"] = float(x[2])
        avg_score, _, _, _ = _cv_score(features, behaviors, task, params, n_splits)
        evaluations.append({**params, "avg_score": avg_score, "source": "refine"})
        return sign * avg_score

    print("\nRefining best parameters with Nelder-Mead simplex...")
    res = minimize(objective, x0, method="Nelder-Mead",
                   options={"maxiter": 60, "xatol": 1e-2, "fatol": 1e-4})

    nm_params = {"C": float(10 ** res.x[0]), "gamma": float(10 ** res.x[1])}
    if refine_epsilon:
        nm_params["epsilon"] = float(max(res.x[2], EPS_MIN))
    nm_score = sign * res.fun

    if task.is_better(nm_score, grid_best_score):
        print(f"Nelder-Mead improved: {nm_params} score {nm_score:.4f} (grid best {grid_best_score:.4f})")
        return nm_params
    print(f"Nelder-Mead did not beat grid best ({grid_best_score:.4f}); keeping grid params.")
    return best_params


def _search(features, behaviors, output_folder, param_grid, task, n_splits, search, suffix="", n_jobs=1):
    """Run the coarse grid, optionally refine with Nelder-Mead, plot the tuning surface,
    and return the chosen best_params."""
    best_params, evaluations = _grid_search(
        features, behaviors, output_folder, param_grid, task, n_splits, suffix=suffix, n_jobs=n_jobs)

    if search == "nelder_mead":
        grid_scores = [e["avg_score"] for e in evaluations]
        grid_best_score = max(grid_scores) if task.greater_is_better else min(grid_scores)
        best_params = _nelder_mead_refine(
            features, behaviors, task, best_params, grid_best_score, n_splits, evaluations)
    elif search != "grid":
        raise ValueError(f"search must be 'grid' or 'nelder_mead', got {search!r}")

    # Resolve sklearn's string gammas ("scale"/"auto") to their numeric value so they can be
    # placed on the tuning plot's log-gamma axis (and flagged as special) instead of dropped.
    for e in evaluations:
        e["gamma_numeric"] = _resolve_numeric_gamma(e["gamma"], features)

    plot_tuning(evaluations, task, output_folder, suffix=suffix)
    return best_params


def _weight_map(est):
    """SVM weight map in input (voxel) space: sum_i (alpha_i * y_i) * support_vector_i, i.e.
    dual_coef_ @ support_vectors_. This is the label-aware map that equals `coef_` for a linear
    kernel and, unlike support_vectors_.mean(), stays non-degenerate when every sample is a
    support vector (strong regularization). Valid for RBF SVR and binary SVC."""
    return (est.dual_coef_ @ est.support_vectors_).ravel()


def _single_permutation(features, behaviors, task, null_params):
    """Run one permutation: shuffle labels, fit SVM, return weight map."""
    perm_behaviors = shuffle(behaviors, random_state=None)
    est = task.make_estimator(null_params)
    est.fit(features, perm_behaviors)
    return _weight_map(est)


def _build_map(features, behaviors, masker, output_folder, task, best_params,
               n_permutations, label="", suffix="", n_jobs=1):
    """Refit at best_params, build the support-vector (beta) map, permutation-test it
    into a z-map, and write all per-map artifacts. Returns a MapResult.

    For one-vs-rest, `behaviors` is the binary (this-class-vs-rest) label vector and
    `suffix` (e.g. "_class1") keeps each class's artifacts distinct."""
    # Train estimator with the best parameters
    est_best = task.make_estimator(best_params)
    est_best.fit(features, behaviors)
    coef_map = _weight_map(est_best)

    n_sv = len(est_best.support_)
    if n_sv == len(features):
        print(f"\nNote: all {n_sv}/{len(features)} samples are support vectors - expected under "
              f"strong regularization (small C); the model's validity rests on the CV score, not "
              f"the SV count. The dual-coef weight map stays label-dependent, so the map is valid.")

    # save model for later predictions
    model_path = output_folder / f'{task.kind}_model{suffix}.pkl'
    save_time = time.time()
    with open(model_path, 'wb') as f:
        pickle.dump(est_best, f)
    print(f"Trained {task.name} model saved to {model_path}, in {easy_time(int(time.time() - save_time))}")

    masker_path = output_folder / 'masker.pkl'
    save_time = time.time()
    with open(masker_path, 'wb') as f:
        pickle.dump(masker, f)
    print(f"Masker saved to {masker_path}, in {easy_time(int(time.time() - save_time))}")

    # saving beta map
    nifti_coef_map = masking.unmask(coef_map, masker)
    nifti_coef_path = output_folder / f'beta_map{suffix}.nii.gz'
    nib.save(nifti_coef_map, nifti_coef_path)

    # Permutation testing
    print("\nPerforming permutation testing...")

    null_params = best_params

    results_file = output_folder / f"null_distributions{suffix}.pkl"

    import os
    actual_jobs = os.cpu_count() if n_jobs == -1 else n_jobs
    print(f"\nRunning {n_permutations} permutations with {actual_jobs} jobs...")
    permute_time = time.time()

    null_maps = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_single_permutation)(features, behaviors, task, null_params)
        for _ in tqdm(range(n_permutations), desc="Queuing permutations")
    )

    elapsed = time.time() - permute_time
    print(f"Permutations completed in {easy_time(int(elapsed))}.")

    # Compute null statistics
    null_stack = np.array(null_maps)
    mean_null = null_stack.mean(axis=0)
    variance = null_stack.var(axis=0, ddof=0)
    std_null = np.sqrt(np.maximum(variance, 0) + 1e-8)
    num_permutations = n_permutations

    # Save null distributions to disk
    with open(results_file, 'wb') as f:
        for wmap in null_maps:
            pickle.dump(wmap, f)
    print(f"Null distribution saved to {results_file}\n")

    del null_maps, null_stack

    # saving null map
    nifti_null_map = masking.unmask(mean_null, masker)
    nifti_null_path = output_folder / f'null_map{suffix}.nii.gz'
    nib.save(nifti_null_map, nifti_null_path)

    # Compute z-map
    zmap = (coef_map - mean_null) / std_null

    # Plot the Histogram. Safety net: if the z-map is entirely zero (a pathological all-null map),
    # skip the histogram rather than crash on min([]) - the z-map + report are still written.
    zmap_flat = zmap[zmap != 0]
    no_signal = zmap_flat.size == 0

    if no_signal:
        print("Warning: z-map has no non-zero values - skipping the z-distribution histogram.")
    else:
        plt.hist(zmap_flat, bins=50, density=True, alpha=0.6, color='blue', label='Z-map distribution')

        # Fit a normal distribution (optional)
        mean, std = np.mean(zmap_flat), np.std(zmap_flat)
        x = np.linspace(min(zmap_flat), max(zmap_flat), 100)
        pdf = norm.pdf(x, mean, std)

        # Overlay the normal distribution
        plt.plot(x, pdf, 'r-', label=f'Normal dist (mu={mean:.2f}, sigma={std:.2f})')

        # Set x-axis to be symmetric around 0
        plt.xlim(left=-max(abs(min(zmap_flat)), abs(max(zmap_flat))), right=max(abs(min(zmap_flat)), abs(max(zmap_flat))))

        # Add labels and legend
        plt.title('Z-Map Distribution')
        plt.xlabel('Z-score')
        plt.ylabel('Density')
        plt.legend()

        plt.savefig(output_folder / f'z_value_distribution{suffix}.png')
        plt.close()
    del zmap_flat

    zmap_threshold_output_folder = output_folder / f"thresholded_zmaps{suffix}"
    Path(zmap_threshold_output_folder).mkdir(parents=True, exist_ok=True)

    # Unmask the z-map back to a 3D image
    print("Unmasking z-map...")
    nifti_zmap = masking.unmask(zmap, masker)
    nifti_zmap_path = output_folder / f'zmap{suffix}.nii.gz'
    nib.save(nifti_zmap, nifti_zmap_path)

    thresholds = [
        (1.644854, 'p05', 'p<0.05'),
        (2.326348, 'p01', 'p<0.01'),
        (2.575829, 'p005', 'p<0.005'),
        (3.090232, 'p001', 'p<0.001')
    ]

    for thresh_val, thresh_label, desc in thresholds:
        print(f"Thresholding z-map at {desc}...")
        nifti_zmap_thresh = threshold_img(nifti_zmap, threshold=thresh_val, cluster_threshold=30)
        nib.save(nifti_zmap_thresh, zmap_threshold_output_folder / f'zmap_{thresh_label}.nii.gz')

    return MapResult(label=label, suffix=suffix, best_params=best_params,
                     coef_map=coef_map, nifti_zmap=nifti_zmap, zmap=zmap, no_signal=no_signal)


def _fit_support_vector_map(features, behaviors, masker, output_folder, param_grid,
                            task, n_permutations, alpha, n_splits, search="grid",
                            label="", suffix="", n_jobs=1):
    """Single decision map (SVR / binary SVC): search hyper-params then build one z-map."""
    best_params = _search(features, behaviors, output_folder, param_grid, task, n_splits,
                          search, suffix=suffix, n_jobs=n_jobs)
    return _build_map(features, behaviors, masker, output_folder, task, best_params,
                      n_permutations, label=label, suffix=suffix, n_jobs=n_jobs)


def _fit_ovr_maps(features, behaviors, masker, output_folder, param_grid,
                  task, n_permutations, alpha, n_splits, search="grid", n_jobs=1):
    """Option A multiclass one-vs-rest: a SINGLE grid search on the full multiclass
    problem (task.scorer, e.g. balanced accuracy) selects shared hyper-parameters, then
    one binary this-class-vs-rest map is built per class at those hyper-parameters -
    yielding K z-maps. Permutation testing (the expensive step) runs once per class."""
    best_params = _search(features, behaviors, output_folder, param_grid, task, n_splits, search, n_jobs=n_jobs)

    classes = np.unique(behaviors)
    print(f"\nOne-vs-rest: building {len(classes)} class maps at shared params {best_params}")

    results = []
    for c in classes:
        c_name = f"class{int(c)}" if float(c).is_integer() else f"class{c}"
        suffix = f"_{c_name}"
        print(f"\n=== One-vs-rest map: {c_name} vs rest ===")
        y_bin = (behaviors == c).astype(int)
        results.append(
            _build_map(features, y_bin, masker, output_folder, task, best_params,
                       n_permutations, label=c_name, suffix=suffix, n_jobs=n_jobs)
        )
    return results


def svm_lsm(features, behaviors, masker, output_folder, param_grid, task,
            n_permutations=1, alpha=0.05, n_splits=5, search="grid", n_jobs=1):
    """
    Generic SVM-based lesion-symptom mapping. Always returns a list of MapResult -
    length 1 for the single-map strategy (SVR and binary SVC), one per class for the
    one-vs-rest multiclass strategy.

    search: "grid" (exhaustive, default) or "nelder_mead" (coarse grid then a
    derivative-free simplex refinement of C/gamma, plus epsilon for SVR).
    """
    if task.map_strategy == "single":
        result = _fit_support_vector_map(
            features, behaviors, masker, output_folder, param_grid,
            task, n_permutations, alpha, n_splits, search=search, n_jobs=n_jobs
        )
        return [result]
    if task.map_strategy == "ovr":
        return _fit_ovr_maps(
            features, behaviors, masker, output_folder, param_grid,
            task, n_permutations, alpha, n_splits, search=search, n_jobs=n_jobs
        )
    raise NotImplementedError(f"map_strategy '{task.map_strategy}' not implemented yet")
