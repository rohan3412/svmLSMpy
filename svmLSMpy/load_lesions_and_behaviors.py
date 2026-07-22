import pandas as pd
import numpy as np
import nibabel as nib
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from nilearn.image import resample_to_img
from .util import normalize_file_name, join_with_and

def load_lesions_and_behaviors(
    lesion_folder,
    csv_file,
    max_score,
    do_regress_out_lesion_volume,
    do_regress_out_covariates,
    output_folder,
    normalize_behavior=True
):
    
    print("Loading behavioral data and lesion files...")
    df = pd.read_csv(csv_file)

    # Validate required columns
    required_columns = ['filename', 'behavior']
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"CSV file must contain '{col}' column.")

    # Match lesion files to CSV filenames
    lesion_files = []
    matched_pairs = []

    for filename in df['filename']:
        norm_filename = normalize_file_name(filename)
        folder_files = os.listdir(lesion_folder)

        # Pass 1: Exact match (compare normalized stems)
        exact_matches = [
            file for file in folder_files
            if normalize_file_name(os.path.splitext(file)[0]) == norm_filename
               or normalize_file_name(file) == norm_filename
        ]

        if len(exact_matches) == 1:
            match = exact_matches[0]
        elif len(exact_matches) > 1:
            raise ValueError(
                f"CSV entry '{filename}' exactly matches multiple lesion files: {exact_matches}"
            )
        else:
            # Pass 2: Substring fallback (bidirectional)
            substring_matches = [
                file for file in folder_files
                if norm_filename in normalize_file_name(file)
                   or normalize_file_name(file) in norm_filename
            ]

            if len(substring_matches) == 1:
                match = substring_matches[0]
                print(f"  Note: '{filename}' matched via substring to '{match}'")
            elif len(substring_matches) > 1:
                raise ValueError(
                    f"CSV entry '{filename}' has no exact match and matches multiple files "
                    f"via substring: {substring_matches}"
                )
            else:
                raise FileNotFoundError(
                    f"No lesion files found in '{lesion_folder}' matching '{filename}' "
                    f"(normalized as '{norm_filename}') by exact or substring match"
                )

        lesion_file = os.path.join(lesion_folder, match)
        if lesion_file in [pair[1] for pair in matched_pairs]:
            prev_entry = next(pair[0] for pair in matched_pairs if pair[1] == lesion_file)
            raise ValueError(
                f"Multiple CSV entries map to the same lesion file:\n"
                f" - '{prev_entry}' and '{filename}' both map to '{lesion_file}'"
            )

        matched_pairs.append((filename, lesion_file))
        lesion_files.append(lesion_file)

    # Process behaviors
    behaviors = df['behavior'].values
    print("\nBehavior before normalization:\n", behaviors)

    #check if scores are binary
    unique_vals = np.unique(behaviors[~np.isnan(behaviors)])
    print(f"\nUnique behavior values: {unique_vals}")
    is_binary = (len(unique_vals) == 2)
    print(f"Behavior is binary: {is_binary}")
    lesion_overlap_groupA = None
    lesion_overlap_groupB = None
    groupA_val = None
    groupB_val = None

    
    if normalize_behavior:
        behaviors = behaviors / max_score
        print("Behavior after normalization:\n", behaviors)
    else:
        print("Behavior normalization skipped (categorical target):\n", behaviors)

    # Display temporary DataFrame for verification
    temp_df = pd.DataFrame({
        "Lesion": lesion_files,
        "Score": df['behavior']
    })
    print("\nTemporary DataFrame (Lesion <-> Score from CSV):")
    print(temp_df.to_string(index=False))
    print(f"\nNumber of lesions: {len(temp_df['Lesion'])}")
    print(f"Number of scores: {len(temp_df['Score'])}\n")

    covariates = None

    # Compute lesion volumes
    lesion_volumes = []
    lesion_imgs = [nib.load(f) for f in lesion_files]
    # Subjects' raw lesion files can differ in voxel grid (shape/affine) - resample
    # onto the first subject's grid before stacking for the group overlap maps below.
    # (Native-resolution volumes are still computed separately further down.)
    reference_img = lesion_imgs[0]
    lesion_data_list = [
        resample_to_img(img, reference_img, interpolation='nearest').get_fdata()
        for img in lesion_imgs
    ]

    if is_binary:
        groupA_val, groupB_val = unique_vals
        print(f"Detected binary behavior values: {groupA_val} and {groupB_val}")
        
        # Indices for each behavior group
        idxA = np.where(behaviors == groupA_val)[0]
        idxB = np.where(behaviors == groupB_val)[0]

        print(f"Group {groupA_val} has {len(idxA)} subjects")
        print(f"Group {groupB_val} has {len(idxB)} subjects")

        # Compute overlap maps by summing the lesion masks for each group
        if len(idxA) > 0:
            lesion_overlap_groupA = np.sum([lesion_data_list[i] > 0 for i in idxA], axis=0)

        if len(idxB) > 0:
            lesion_overlap_groupB = np.sum([lesion_data_list[i] > 0 for i in idxB], axis=0)

        # Save overlap maps as NIfTI files
        if lesion_overlap_groupA is not None:
            overlap_imgA = nib.Nifti1Image(lesion_overlap_groupA.astype(np.uint8), lesion_imgs[0].affine)
            overlap_filenameA = output_folder / f"lesion_overlap_group_{groupA_val}.nii.gz"
            nib.save(overlap_imgA, overlap_filenameA)
            print(f"Lesion overlap map for group {groupA_val} saved as {overlap_filenameA}")

        if lesion_overlap_groupB is not None:
            overlap_imgB = nib.Nifti1Image(lesion_overlap_groupB.astype(np.uint8), lesion_imgs[0].affine)
            overlap_filenameB = output_folder / f"lesion_overlap_group_{groupB_val}.nii.gz"
            nib.save(overlap_imgB, overlap_filenameB)
            print(f"Lesion overlap map for group {groupB_val} saved as {overlap_filenameB}")
    
    # Compute lesion volumes as before
    for file in tqdm(lesion_files, desc="Loading lesions and computing volumes"):
        lesion_img = nib.load(file)
        lesion_data = lesion_img.get_fdata()
        voxel_volume = np.prod(lesion_img.header.get_zooms())
        volume = np.sum(lesion_data > 0) * voxel_volume
        lesion_volumes.append(volume)

    lesion_volumes = np.array(lesion_volumes).reshape(-1, 1)
    print(
        f"Lesion volumes summary: min={np.nanmin(lesion_volumes):.3f}, "
        f"max={np.nanmax(lesion_volumes):.3f}, NaNs={np.isnan(lesion_volumes).sum()}"
    )

    # Empty-input guard: fail fast if the maps carry no data. An SVM trained on all-zero
    # features only ever returns chance scores, all-support-vector degeneracy and empty
    # beta/z-maps - so catch it here (in seconds) rather than after grid search + permutations.
    n_empty = int(np.sum(lesion_volumes == 0))
    if n_empty == len(lesion_files):
        raise ValueError(
            f"All {len(lesion_files)} input maps in '{lesion_folder}' are empty (every voxel is "
            f"zero) - there is nothing to analyze. Check your lesion/disconnectome files; a "
            f"failed export commonly writes ~all-zero volumes (an empty 181x217x181 map gzips "
            f"to only ~27 KB)."
        )
    if n_empty:
        print(f"WARNING: {n_empty}/{len(lesion_files)} input maps are empty (all-zero) and "
              f"contribute no features.")

    # Process covariates and lesion volumes only if do_regress_out_covariates is True
    covariate_names = None
    if do_regress_out_covariates:
        # Load covariates from CSV
        covariates = df.drop(columns=["filename", "behavior"]).select_dtypes(include=["number"])
        has_covariates = covariates.shape[1] > 0
        covariate_names = []

        # Handle covariates and lesion volumes
        if has_covariates:
            # Handle NaNs
            if covariates.isna().any().any():
                covariates = covariates.fillna(covariates.mean())
                print("Filled NaN values in covariates with column means.")
            covariate_names = list(covariates.columns)
            covariates = np.array(covariates)  # Convert to NumPy array for consistency

        if do_regress_out_lesion_volume:
            if has_covariates:
                print(f"Loaded {covariates.shape[1]} additional covariates ({join_with_and(covariate_names)}) and lesion volume as covariate.")
                covariates = np.hstack([lesion_volumes, covariates])
            else:
                print("No additional covariates found in the CSV, using only lesion volume as covariate.")
                covariates = lesion_volumes
            covariate_names = ["lesion_volume"] + covariate_names
        else:
            # No lesion volumes
            if has_covariates:
                print(f"Loaded {covariates.shape[1]} additional covariates ({join_with_and(covariate_names)}).")
            else:
                print("No additional covariates found in the CSV (and regress_out_lesion_volume is False).")

        # Z-transform covariates
        if covariates is not None:
            covariates = StandardScaler().fit_transform(covariates)
    else:
        print("SKIPPED Covariate processing and lesion volume computation.")

    return lesion_files, behaviors, covariates, lesion_volumes, covariate_names





