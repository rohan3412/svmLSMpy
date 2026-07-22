import nibabel as nib
import numpy as np
from sklearn.preprocessing import normalize
from nilearn import masking
from tqdm import tqdm
from nilearn.image import resample_to_img
import time

def filter_voxels_by_patient_count(lesion_files, min_patient_count, normalize_vector, output_folder):
    """
    Filter voxels by the number of patients they are involved in.
    """
    no_of_patients = len(lesion_files)
    if not isinstance(min_patient_count, int):
        if isinstance(min_patient_count, str):
            if min_patient_count.endswith('%'):
                min_patient_count = min_patient_count.strip('%')
            min_patient_count = float(min_patient_count)
            print(f"{min_patient_count}% of {no_of_patients} patients = ",end='')
            
            min_patient_count = no_of_patients*min_patient_count/100
            min_patient_count = round(min_patient_count)
            
            print(min_patient_count)
            print("Thus,")


    if min_patient_count>0:
        print(f"Filtering voxels by patient count: {min_patient_count}/{no_of_patients}")
    else:
        print("Filtering is not done")
    '''
    lesion_imgs = [nib.load(f) for f in tqdm(lesion_files, desc="Loading lesion images")]
    masker = masking.compute_brain_mask(lesion_imgs[0])
    lesion_data = [masking.apply_mask(img, masker) for img in tqdm(lesion_imgs, desc="Applying mask to lesion images")]
    '''
    time.sleep(1)
    lesion_imgs = [nib.load(f) for f in tqdm(lesion_files, desc="Loading lesion images")]
    reference_img = lesion_imgs[0]
    masker = masking.compute_brain_mask(reference_img)
    lesion_imgs_resampled = [
        resample_to_img(img, reference_img, interpolation='nearest') 
        for img in tqdm(lesion_imgs, desc="Resampling lesion images")
    ]
    lesion_data = [
        masking.apply_mask(img, masker) 
        for img in tqdm(lesion_imgs_resampled, desc="Applying mask to lesion images")
    ]
    
    # Stack the vectorized lesion data
    lesion_data_stack = np.vstack(lesion_data)

    sum_of_vectors = np.sum(lesion_data_stack, axis=0)
    sum_of_vectors = sum_of_vectors.astype(np.int32)  # Ensure integer type
    sum_of_voxel_mni = masking.unmask(sum_of_vectors, masker)
    sum_of_vectors_path = output_folder / "lesion_overlap.nii.gz"
    nib.save(sum_of_voxel_mni, sum_of_vectors_path)

    # Count the number of patients each voxel is involved in
    voxel_patient_count = np.sum(lesion_data_stack > 0, axis=0)

    # Filter voxels: drop those below minimum count, and those damaged in 100% of patients (0 variance)
    valid_mask = (voxel_patient_count >= min_patient_count) & (voxel_patient_count < no_of_patients)

    lesion_data_prepared = lesion_data_stack[:, valid_mask]
    
    # Update the masker so downstream unmasking expects the reduced feature size
    new_masker = masking.unmask(valid_mask.astype(np.int8), masker)

    sum_of_vectors_filtered = np.sum(lesion_data_prepared, axis=0)
    sum_of_vectors_filtered = sum_of_vectors_filtered.astype(np.int32)  # Ensure integer type
    sum_of_voxel_mni_filtered = masking.unmask(sum_of_vectors_filtered, new_masker)
    sum_of_vectors_filtered_path = output_folder / "lesion_overlap_filtered.nii.gz"
    nib.save(sum_of_voxel_mni_filtered, sum_of_vectors_filtered_path)

    if normalize_vector:
        # Normalize the data to have unit norm
        lesion_data_prepared = normalize(lesion_data_prepared, norm='l2', axis=1)

    print(f"Features reduced from {lesion_data_stack.shape[1]} to {lesion_data_prepared.shape[1]} voxels.")

    return min_patient_count, lesion_data_prepared, new_masker







