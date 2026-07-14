from sklearn.linear_model import LinearRegression
import numpy as np
from tqdm import tqdm

def regress_covariates_from_behavior(behaviors, covariates):
    # Regress covariates out of behavioral scores.
    if covariates is None:
        print('NO covariates')
        return behaviors
    print("Regressing covariates out of behavioral scores...")
    lr = LinearRegression()
    lr.fit(covariates, behaviors)
    residuals = behaviors - lr.predict(covariates)
    print('\nResiduals:\n', residuals,"\n")
    return residuals

def regress_covariates_from_lesions(features, covariates):
    if covariates is None:
        print('NO covariates to regress from lesion data')
        return features

    # Initialize output array
    residualized_features = np.zeros_like(features)

    # For each voxel, regress out the covariates
    n_voxels = features.shape[1]
    lr = LinearRegression()

    for voxel_idx in tqdm(range(n_voxels), desc="Regressing covariates out of lesion data..."):
        voxel_values = features[:, voxel_idx]

        # Fit linear regression: voxel_values ~ covariates
        lr.fit(covariates, voxel_values)

        # Calculate residuals: observed - predicted
        residuals = voxel_values - lr.predict(covariates)

        # Store residualized voxel values
        residualized_features[:, voxel_idx] = residuals

    print(f"Regressed covariates from {n_voxels} voxels")

    return residualized_features



