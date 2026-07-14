import numpy as np
import nibabel as nib
from nilearn.image import load_img
from nilearn.plotting import plot_img

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

bg_img_path = files("svmLSMpy.resources").joinpath("mni152.nii.gz")
bg_img = str(bg_img_path)

# Axis configuration for different slice orientations
AXIS_CONFIG = {
    'axial': {'dim': 2, 'axis_tuple': (0, 1), 'voxel_template': [0, 0, None], 'display_mode': 'z'},
    'coronal': {'dim': 1, 'axis_tuple': (0, 2), 'voxel_template': [0, None, 0], 'display_mode': 'y'},
    'sagittal': {'dim': 0, 'axis_tuple': (1, 2), 'voxel_template': [None, 0, 0], 'display_mode': 'x'}
}


def get_slice_coordinates(nii_file_path, num_slices, orientation='axial', adjustment=3.0):
    if orientation not in AXIS_CONFIG:
        raise ValueError(f"Orientation must be one of {list(AXIS_CONFIG.keys())}")
    
    config = AXIS_CONFIG[orientation]
    img = load_img(nii_file_path)
    affine = img.affine
    data = img.get_fdata()
    
    # Find non-empty slices along the specified axis
    non_empty_slices = np.any(data > 0, axis=config['axis_tuple'])
    non_empty_indices = np.where(non_empty_slices)[0]
    
    if len(non_empty_indices) == 0:
        raise ValueError("The NIfTI image has no non-empty slices.")
    
    min_index, max_index = non_empty_indices[0], non_empty_indices[-1]
    
    # Convert voxel indices to world coordinates
    voxel_min = config['voxel_template'].copy()
    voxel_max = config['voxel_template'].copy()
    voxel_min[config['dim']] = min_index
    voxel_max[config['dim']] = max_index
    
    world_coords = nib.affines.apply_affine(affine, [voxel_min, voxel_max])
    coord_min, coord_max = world_coords[:, config['dim']]
    
    # Adjust bounds
    coord_min_adjusted = coord_min - adjustment
    coord_max_adjusted = coord_max + adjustment
    
    # Generate evenly spaced cut coordinates
    cut_coords = np.linspace(coord_min_adjusted, coord_max_adjusted, num_slices).tolist()
    
    print(f"{orientation.capitalize()} cut slice coordinates:", cut_coords)
    
    return cut_coords


def save_slice_mosaic(nii_file_path, cut_coords, output_image_path, 
                      orientation='axial', max_activation=None, 
                      cmap='jet', threshold=0, black_bg=False, colorbar=True):
    if orientation not in AXIS_CONFIG:
        raise ValueError(f"Orientation must be one of {list(AXIS_CONFIG.keys())}")
    
    img = load_img(nii_file_path)
    display_mode = AXIS_CONFIG[orientation]['display_mode']
    
    vmin = -max_activation if max_activation is not None else None
    vmax = max_activation if max_activation is not None else None
    
    plot_img(img, cut_coords, output_image_path, 
             display_mode=display_mode, threshold=threshold, 
             bg_img=bg_img, vmin=vmin, vmax=vmax, 
             black_bg=black_bg, colorbar=colorbar, cmap=cmap)


