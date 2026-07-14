# svmLSMpy

Python toolkit for **Support Vector Lesion-Symptom Mapping**, covering both
**Support Vector Regression (SVR)** for continuous behavioural scores and
**Support Vector Classification (SVC)** for categorical outcomes (binary and
multiclass), with permutation-based z-maps, visualization, and automated HTML reports.

It automatically routes to the right pipeline based on the behavioural score, sharing
one code path for loading, voxel filtering, covariate handling, permutation testing,
and reporting.

## Install

```
pip install -e .
```

## Getting started

Make a folder of binary lesion files (`.nii` / `.nii.gz`) and a CSV with two mandatory
columns:

- `filename` — the lesion filename (with extension) for each subject
- `behavior` — the behavioural score (continuous → SVR; categorical → SVC)

Any additional numeric columns are treated as covariates.

### Automatic routing (recommended)

```python
from svmLSMpy import run_svm_lsm

run_svm_lsm(
    symptom_folder="lesions/",
    csv_path="behavior.csv",
    output_path="output/",
    mode="auto",        # "svr" or "svc" to override the detection
    max_score=37,       # required only for the SVR (continuous) path
)
```

`mode="auto"` inspects the `behavior` column:

- non-numeric, or ≤2 distinct values, or integer-valued with ≤10 distinct levels → **SVC**
  (binary if 2 classes, one-vs-rest multiclass otherwise)
- otherwise → **SVR**

The chosen path is printed loudly; pass `mode="svr"` / `mode="svc"` to force it.

## How SVR and SVC differ

| | SVR | SVC |
|---|---|---|
| Behaviour | continuous | categorical (binary / multiclass) |
| Estimator | `SVR` (rbf) | `SVC` (rbf, `class_weight="balanced"`) |
| Cross-validation | `KFold` | `StratifiedKFold` |
| Model selection | mean squared error | ROC-AUC (binary) / balanced accuracy (multiclass) |
| Covariates | regressed out of behaviour **and** lesions | lesion (feature) side only¹ |
| Output maps | one z-map | one z-map (binary) / one per class, one-vs-rest (multiclass) |

¹ Covariates are never regressed out of categorical labels (statistically incoherent);
for SVC they are removed feature-side only, which is behaviour-agnostic and valid for
both continuous and categorical targets.

## Tests

Synthetic fixtures (no real data needed) exercise every path end-to-end:

```
python tests/make_fixture.py          # generate synthetic lesions + behaviour CSVs
python tests/verify_svr.py            # SVR reproduces the golden baseline
python tests/verify_svc_binary.py     # binary SVC
python tests/verify_svc_multiclass.py # multiclass one-vs-rest
python tests/verify_routing.py        # auto-routing + overrides
```
