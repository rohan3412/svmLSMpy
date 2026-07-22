from .util import get_current_datetime_for_filename, easy_time
from .load_lesions_and_behaviors import load_lesions_and_behaviors
from .filter_voxels_by_patient_count import filter_voxels_by_patient_count
from .regress_covariates import regress_covariates_from_behavior, regress_covariates_from_lesions
from .covariate_correlation import save_correlation_matrix
from .svm_lsm import svm_lsm
from .save_report import save_report
from .task import SVR_TASK, SVC_BINARY_TASK, SVC_MULTICLASS_TASK

import time
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from nilearn import masking
import nibabel as nib

# A numeric behaviour with more than this many distinct integer levels is treated as
# continuous (SVR) rather than categorical (SVC) during auto-routing.
MULTICLASS_MAX_CLASSES = 10


def _resolve_task(mode, csv_path):
    """Pick the LSMTask from `mode` ("auto" | "svr" | "svc") and the behaviour column.

    auto: non-numeric OR <=2 distinct values OR integer-valued with <=MULTICLASS_MAX
    distinct levels -> SVC (categorical); otherwise SVR (continuous). Binary vs
    one-vs-rest multiclass is decided by the number of classes. The choice is printed
    loudly - pass mode="svr"/"svc" to override the detection.
    """
    behavior = pd.read_csv(csv_path)["behavior"].dropna()
    n_classes = int(behavior.nunique())

    if mode not in ("auto", "svr", "svc"):
        raise ValueError(f"mode must be 'auto', 'svr' or 'svc', got {mode!r}")

    if mode == "svr":
        categorical, reason = False, "forced by mode='svr'"
    elif mode == "svc":
        categorical, reason = True, "forced by mode='svc'"
    else:  # auto
        if not is_numeric_dtype(behavior):
            categorical, reason = True, "behaviour column is non-numeric"
        elif n_classes <= 2:
            categorical, reason = True, f"only {n_classes} distinct values"
        else:
            vals = behavior.to_numpy(dtype=float)
            int_like = np.allclose(vals, np.round(vals))
            if int_like and n_classes <= MULTICLASS_MAX_CLASSES:
                categorical, reason = True, f"integer-valued with {n_classes} distinct levels"
            else:
                categorical, reason = False, f"continuous ({n_classes} distinct values)"

    if not categorical:
        task = SVR_TASK
    else:
        task = SVC_BINARY_TASK if n_classes <= 2 else SVC_MULTICLASS_TASK

    print("\n" + "=" * 70)
    print(f"[svmLSMpy] routing (mode={mode!r}): {reason}")
    print(f"[svmLSMpy] -> {task.name}   (override with mode='svr' or mode='svc')")
    print("=" * 70 + "\n")
    return task


def run_svm_lsm(symptom_folder,
                csv_path,
                output_path,
                mode="auto",
                max_score=None,
                behaviour_name="behavioural_deficit",
                regress_out_lesion_volume=True,
                regress_out_covariates_on_scores=True,
                regress_out_covariates_on_lesions=True,
                normalize_vector=False,
                min_patient_count='10%',
                param_grid=None,
                n_permutations=1000,
                alpha=0.05,
                n_splits=5,
                num_slices=7,
                search="grid",
                n_jobs=-1):
    """
    Unified SVM-LSM entry point. Auto-detects SVR (continuous behaviour) vs SVC
    (categorical behaviour, binary or one-vs-rest multiclass) from the 'behavior'
    column, or is forced with mode="svr"/"svc".

    max_score is required only for SVR (the behaviour is divided by it); it is ignored
    for SVC.

    search selects hyper-parameter tuning: "grid" (exhaustive, default) or "nelder_mead"
    (coarse grid then a derivative-free simplex refinement of C/gamma, plus epsilon for SVR).
    """
    task = _resolve_task(mode, csv_path)

    if task.normalize_behavior and max_score is None:
        raise ValueError("max_score is required for SVR (continuous behaviour normalisation)")

    return _run_with_task(symptom_folder,
                          csv_path,
                          max_score,
                          output_path,
                          task,
                          behaviour_name=behaviour_name,
                          regress_out_lesion_volume=regress_out_lesion_volume,
                          regress_out_covariates_on_scores=regress_out_covariates_on_scores,
                          regress_out_covariates_on_lesions=regress_out_covariates_on_lesions,
                          normalize_vector=normalize_vector,
                          min_patient_count=min_patient_count,
                          param_grid=param_grid,
                          n_permutations=n_permutations,
                          alpha=alpha,
                          n_splits=n_splits,
                          num_slices=num_slices,
                          search=search,
                          n_jobs=n_jobs)


def _run_with_task(symptom_folder,
                   csv_path,
                   max_score,
                   output_path,
                   task,
                   behaviour_name="behavioural_deficit",
                   regress_out_lesion_volume=True,
                   regress_out_covariates_on_scores=True,
                   regress_out_covariates_on_lesions=True,
                   normalize_vector=False,
                   min_patient_count='10%',
                   param_grid=None,
                   n_permutations=1000,
                   alpha=0.05,
                   n_splits=5,
                   num_slices=7,
                   search="grid",
                   n_jobs=-1):
    """
    Shared SVM-LSM orchestrator for both SVR and SVC. Loads lesions + behaviour,
    filters voxels, regresses covariates, runs the generic svm_lsm core (which returns
    one MapResult per decision map), then writes one HTML report per map.

    `task` (SVR_TASK / SVC_BINARY_TASK / SVC_MULTICLASS_TASK) carries everything that
    differs between pipelines; called via the public run_svm_lsm entry point.
    """
    start_time = time.time()

    if param_grid is None:
        param_grid = task.default_grid

    symptom = behaviour_name
    output_folder = Path(f"{output_path}/{symptom}_{n_permutations}_results_{get_current_datetime_for_filename()}")
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load lesions and behaviors
    lesion_folder = Path(symptom_folder)

    # Behaviour-side covariate regression is only valid for a continuous target (SVR).
    # For SVC it is skipped entirely - covariates are removed feature-side instead.
    do_behavior_side = task.behavior_regression and regress_out_covariates_on_scores
    do_regress_out_covariates = do_behavior_side or regress_out_covariates_on_lesions

    lesion_files, behaviors, covariates, lesion_volumes, covariate_names = load_lesions_and_behaviors(
        lesion_folder, csv_path, max_score, regress_out_lesion_volume,
        do_regress_out_covariates, output_folder,
        normalize_behavior=task.normalize_behavior)
    print("\n\tTIME ELAPSED : ", easy_time(int(time.time() - start_time)), end="\n\n")

    save_correlation_matrix(behaviors, task, covariate_names, covariates, lesion_volumes, output_folder)

    min_patient_count, features, masker = filter_voxels_by_patient_count(
        lesion_files, min_patient_count, normalize_vector, output_folder)

    if covariates is not None:
        if do_behavior_side:
            behaviors = regress_covariates_from_behavior(behaviors, covariates)
            print("\n\tTIME ELAPSED : ", easy_time(int(time.time() - start_time)), end="\n\n")
        else:
            print("covariates not regressed from behavioral score")

        if regress_out_covariates_on_lesions:
            non_regressed_features = features
            try:
                print("Running... Press Ctrl+C to stop")
                features = regress_covariates_from_lesions(features, covariates)
                nib.save(masking.unmask(np.sum(features, axis=0), masker),
                         output_folder / 'covariate_regressed_out_of_overlap_map.nii.gz')
                print("\n\tTIME ELAPSED : ", easy_time(int(time.time() - start_time)), end="\n\n")

            except KeyboardInterrupt:
                print("Lesion file covariate regression cancelled by user!")
                print("\n" * 7)
                regress_out_covariates_on_lesions = False
                features = non_regressed_features
        else:
            print("covariates not regressed from lesion file")
    else:
        print("\n\nNo covariates present\n\n")

    # Core: one MapResult per decision map (SVR / binary SVC -> 1; one-vs-rest -> K)
    results = svm_lsm(features=features,
                      behaviors=behaviors,
                      masker=masker,
                      output_folder=output_folder,
                      param_grid=param_grid,
                      task=task,
                      n_permutations=n_permutations,
                      alpha=alpha,
                      n_splits=n_splits,
                      search=search,
                      n_jobs=n_jobs)

    # Dataset statistics
    num_lesions = len(lesion_files)
    num_patients = len(behaviors)
    mean_lesion_volume = np.mean(lesion_volumes)
    time_taken = easy_time(time.time() - start_time)

    # Print for covariate information
    if covariates is None:
        print("covariates is None")
    else:
        print(f"covariates={covariates}")

    # Report tail: one report per decision map. suffix keeps one-vs-rest maps distinct.
    for r in results:
        label_suffix = f" ({r.label})" if r.label else ""
        signal_suffix = " - NO SIGNAL (all-zero z-map)" if r.no_signal else ""
        report_path = output_folder / f"{task.kind}_lsm_report{r.suffix}.html"
        save_report(report_path,
                    r.best_params,
                    behaviour_name + label_suffix + signal_suffix,
                    n_permutations,
                    alpha,
                    r.zmap,
                    min_patient_count,
                    num_patients,
                    num_slices,
                    r.nifti_zmap,
                    time_taken,
                    num_lesions,
                    mean_lesion_volume,
                    covariates,
                    regress_out_lesion_volume,
                    regress_out_covariates_on_scores,
                    regress_out_covariates_on_lesions,
                    normalize_vector,
                    model_name=task.report_name,
                    suffix=r.suffix)

    print("\n\tTOTAL TIME TAKEN : ", easy_time(int(time.time() - start_time)))
    return results
