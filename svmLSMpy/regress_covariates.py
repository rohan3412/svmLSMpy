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

    print("Regressing covariates out of lesion data (vectorized)...")
    n_voxels = features.shape[1]

    # Add intercept column to covariates: shape (N, K+1)
    X = np.column_stack([np.ones(covariates.shape[0]), covariates])

    # Solve all voxels at once: beta = (X'X)^-1 X'Y where Y is (N, V)
    beta, _, _, _ = np.linalg.lstsq(X, features, rcond=None)

    # Residuals = observed - predicted
    residualized_features = features - X @ beta

    print(f"Regressed covariates from {n_voxels} voxels")
    return residualized_features
