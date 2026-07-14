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
from sklearn.utils import shuffle
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

from .util import easy_time, easy_eta


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


def _grid_search(features, behaviors, output_folder, param_grid, task, n_splits, suffix=""):
    """Grid-search hyper-parameters by K-fold CV using the task's scorer. Writes
    results_and_scores{suffix}.csv and returns the best parameter dict."""
    print(f"Running {task.name} analysis...")

    # Perform grid search with K-fold cross-validation
    print("Performing grid search with K-fold cross-validation...")

    param_names = list(task.param_names)
    param_combinations = list(product(*[param_grid[k] for k in param_names]))

    best_params = dict(zip(param_names, param_combinations[0]))
    best_score = task.worst_score()
    best_iteration = 1

    patient_count = len(features)
    all_scores = []
    i = 1
    num_iter = len(param_combinations)

    for idx, combo in enumerate(param_combinations, 1):
        params = dict(zip(param_names, combo))
        print(f"\nIteration: {idx}/{num_iter}, Testing parameters: {params}")
        iter_time = time.time()

        scores = []
        no_of_sv = []

        cv = task.make_cv(n_splits)
        splits = cv.split(features, behaviors) if task.stratified else cv.split(features)

        for fold_idx, (train_idx, test_idx) in enumerate(splits, 1):
            print(f"Split :{fold_idx}/{n_splits}")
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = behaviors[train_idx], behaviors[test_idx]

            est = task.make_estimator(params)
            est.fit(X_train, y_train)

            print(f"\tno. of support vectors : {len(est.support_)}/{patient_count}", )
            no_of_sv.append(len(est.support_))

            y_pred = est.decision_function(X_test) if task.score_uses_decision else est.predict(X_test)
            score = task.scorer(y_test, y_pred)

            print(f"\tscore : {score}", )
            scores.append(score)

        # Average score across all folds
        avg_score = np.mean(scores)
        avg_no_of_sv = np.mean(no_of_sv)

        all_scores.append((i, *combo, avg_score, scores, avg_no_of_sv, no_of_sv))

        print(f"\nAverage score: {avg_score:.4f}")
        print(scores, "\n")

        # Update best parameters if current score is better
        if task.is_better(avg_score, best_score):
            best_iteration = i
            best_score = avg_score
            best_params = params

        i = i + 1
        print(f"Best iteration: {best_iteration}, Best score: {best_score:.4f}, Current Score: {avg_score:.4f}")
        print(f"Iteration {best_iteration} parameters: {best_params}")

        print(f"Iteration time: {easy_time(int(time.time() - iter_time))}")

    columns = ["Iteration"] + [n.capitalize() for n in param_names] + [
        "Avg_Score", "Scores", "Avg_Support_Vectors", "Support_Vectors"
    ]

    df = pd.DataFrame(all_scores, columns=columns)

    # Save to CSV
    output_path = output_folder / f'results_and_scores{suffix}.csv'
    df.to_csv(output_path, index=False)
    print(f"Results and scores saved to {output_path}")

    print(f"\nBest parameters found: {best_params} with score {best_score:.4f} in iteration {best_iteration}/{num_iter}")
    return best_params


def _build_map(features, behaviors, masker, output_folder, task, best_params,
               n_permutations, label="", suffix=""):
    """Refit at best_params, build the support-vector (beta) map, permutation-test it
    into a z-map, and write all per-map artifacts. Returns a MapResult.

    For one-vs-rest, `behaviors` is the binary (this-class-vs-rest) label vector and
    `suffix` (e.g. "_class1") keeps each class's artifacts distinct."""
    # Train estimator with the best parameters
    est_best = task.make_estimator(best_params)
    est_best.fit(features, behaviors)
    coef_map = est_best.support_vectors_.mean(axis=0)

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

    sum_null = None
    sum_null_squared = None
    num_permutations = 0

    with open(results_file, 'wb') as f:
        permute_time = time.time()

        time.sleep(1)
        with tqdm(range(n_permutations), desc="Running permutations", unit="permutation", mininterval=1, ncols=100, dynamic_ncols=True, leave=True) as pbar:
            for i in pbar:
                perm_behaviors = shuffle(behaviors, random_state=None)

                est_permutation = task.make_estimator(null_params)

                est_permutation.fit(features, perm_behaviors)

                vector_mean = est_permutation.support_vectors_.mean(axis=0)

                pickle.dump(vector_mean, f)

                if sum_null is None:
                    sum_null = np.zeros_like(vector_mean)
                    sum_null_squared = np.zeros_like(vector_mean)

                sum_null += vector_mean
                sum_null_squared += vector_mean ** 2
                num_permutations += 1

                del vector_mean

                elapsed_time = time.time() - permute_time
                pbar.set_postfix(elapsed=f"{easy_time(elapsed_time)}", eta=f"{easy_eta((elapsed_time / (i + 1)) * (n_permutations - i - 1))}")

    print(f"Permutations completed. Null distribution saved to {results_file}\n")

    mean_null = sum_null / num_permutations

    variance = (sum_null_squared / num_permutations) - (mean_null ** 2)
    std_null = np.sqrt(np.maximum(variance, 0) + 1e-8)

    # saving null map
    nifti_null_map = masking.unmask(mean_null, masker)
    nifti_null_path = output_folder / f'null_map{suffix}.nii.gz'
    nib.save(nifti_null_map, nifti_null_path)

    # Compute z-map
    zmap = (coef_map - mean_null) / std_null

    # Plot the Histogram
    zmap_flat = zmap[zmap != 0]

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
                     coef_map=coef_map, nifti_zmap=nifti_zmap, zmap=zmap)


def _fit_support_vector_map(features, behaviors, masker, output_folder, param_grid,
                            task, n_permutations, alpha, n_splits, label="", suffix=""):
    """Single decision map (SVR / binary SVC): grid-search then build one z-map."""
    best_params = _grid_search(features, behaviors, output_folder, param_grid, task, n_splits, suffix=suffix)
    return _build_map(features, behaviors, masker, output_folder, task, best_params,
                      n_permutations, label=label, suffix=suffix)


def _fit_ovr_maps(features, behaviors, masker, output_folder, param_grid,
                  task, n_permutations, alpha, n_splits):
    """Option A multiclass one-vs-rest: a SINGLE grid search on the full multiclass
    problem (task.scorer, e.g. balanced accuracy) selects shared hyper-parameters, then
    one binary this-class-vs-rest map is built per class at those hyper-parameters -
    yielding K z-maps. Permutation testing (the expensive step) runs once per class."""
    best_params = _grid_search(features, behaviors, output_folder, param_grid, task, n_splits)

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
                       n_permutations, label=c_name, suffix=suffix)
        )
    return results


def svm_lsm(features, behaviors, masker, output_folder, param_grid, task,
            n_permutations=1, alpha=0.05, n_splits=5):
    """
    Generic SVM-based lesion-symptom mapping. Always returns a list of MapResult -
    length 1 for the single-map strategy (SVR and binary SVC), one per class for the
    one-vs-rest multiclass strategy.
    """
    if task.map_strategy == "single":
        result = _fit_support_vector_map(
            features, behaviors, masker, output_folder, param_grid,
            task, n_permutations, alpha, n_splits,
        )
        return [result]
    if task.map_strategy == "ovr":
        return _fit_ovr_maps(
            features, behaviors, masker, output_folder, param_grid,
            task, n_permutations, alpha, n_splits,
        )
    raise NotImplementedError(f"map_strategy '{task.map_strategy}' not implemented yet")
