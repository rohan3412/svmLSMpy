"""
LSMTask - the small strategy object that captures everything that differs between
the SVR and SVC lesion-symptom-mapping pipelines. The generic core in svm_lsm.py
reads these fields instead of hard-coding SVR; the orchestrator picks a task by
routing on the behavioural score.

Only four things genuinely diverge between SVR and SVC:
  1. the estimator (SVR vs SVC, epsilon vs class_weight)
  2. the hyper-parameter names in the grid (epsilon vs class_weight)
  3. the cross-validation splitter (KFold vs StratifiedKFold)
  4. the scoring metric + its optimisation direction (MSE-min vs AUC/bal-acc-max)

Everything else (support-vector map, permutation null, z-map, thresholds) is shared.
"""
from dataclasses import dataclass, field
from typing import Callable, Sequence

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, roc_auc_score, balanced_accuracy_score


@dataclass
class LSMTask:
    name: str                          # human label, e.g. "SVR", "SVC (binary)"
    kind: str                          # "svr" | "svc"
    param_names: Sequence[str]         # grid keys, in grid-iteration order
    make_estimator: Callable           # (params: dict) -> fitted-ready sklearn estimator
    make_cv: Callable                  # (n_splits: int) -> CV splitter
    scorer: Callable                   # (y_true, y_pred_or_scores) -> float
    greater_is_better: bool            # optimisation direction for the scorer
    stratified: bool = False           # whether cv.split needs y
    score_uses_decision: bool = False  # feed decision_function output to scorer (else predict)
    behavior_regression: bool = True   # regress covariates out of the behaviour (SVR only)
    map_strategy: str = "single"       # "single" (SVR / binary SVC) | "ovr" (multiclass SVC)
    normalize_behavior: bool = True    # divide behaviour by max_score (continuous SVR only)
    report_name: str = "Support Vector Machine"  # method name shown in the HTML report
    default_grid: dict = field(default_factory=dict)  # grid used when caller passes none

    def worst_score(self) -> float:
        """Initial 'best' value so the first candidate always wins."""
        return float("-inf") if self.greater_is_better else float("inf")

    def is_better(self, candidate: float, incumbent: float) -> bool:
        return candidate > incumbent if self.greater_is_better else candidate < incumbent


# Log-spaced defaults. C spans strong->weak regularization; gamma stays small (large gamma
# forces every sample to become a support vector -> overfit/degenerate map) and brackets
# sklearn's data-dependent "scale"/"auto". Nelder-Mead (search="nelder_mead") refines from here.
_C_GRID = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
_GAMMA_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, "scale", "auto"]

# --- SVR: the original pipeline, now expressed as a task ---------------------
SVR_TASK = LSMTask(
    name="SVR",
    kind="svr",
    param_names=("C", "gamma", "epsilon"),
    make_estimator=lambda p: SVR(kernel="rbf", C=p["C"], gamma=p["gamma"], epsilon=p["epsilon"]),
    make_cv=lambda n: KFold(n_splits=n, shuffle=True, random_state=42),
    scorer=mean_squared_error,
    greater_is_better=False,
    stratified=False,
    score_uses_decision=False,
    behavior_regression=True,
    map_strategy="single",
    normalize_behavior=True,
    report_name="Support Vector Regression (SVR)",
    default_grid={"C": _C_GRID, "gamma": _GAMMA_GRID, "epsilon": [0.1]},
)

# --- Binary SVC: one decision map, ROC-AUC model selection -------------------
SVC_BINARY_TASK = LSMTask(
    name="SVC (binary)",
    kind="svc",
    param_names=("C", "gamma"),
    make_estimator=lambda p: SVC(kernel="rbf", C=p["C"], gamma=p["gamma"], class_weight="balanced"),
    make_cv=lambda n: StratifiedKFold(n_splits=n, shuffle=True, random_state=42),
    scorer=roc_auc_score,               # threshold-free, reads decision_function scores
    greater_is_better=True,
    stratified=True,
    score_uses_decision=True,
    behavior_regression=False,          # never residualise categorical labels
    map_strategy="single",
    normalize_behavior=False,           # labels stay as class integers
    report_name="Support Vector Classification (SVC)",
    default_grid={"C": _C_GRID, "gamma": _GAMMA_GRID},
)

# --- Multiclass SVC: one grid search (balanced accuracy), then one-vs-rest maps ---
# Model selection uses balanced accuracy on the full multiclass problem; the
# one-vs-rest decomposition into K z-maps is performed by the core (map_strategy="ovr").
SVC_MULTICLASS_TASK = LSMTask(
    name="SVC (multiclass, one-vs-rest)",
    kind="svc",
    param_names=("C", "gamma"),
    make_estimator=lambda p: SVC(kernel="rbf", C=p["C"], gamma=p["gamma"], class_weight="balanced"),
    make_cv=lambda n: StratifiedKFold(n_splits=n, shuffle=True, random_state=42),
    scorer=balanced_accuracy_score,     # robust to imbalance, defined for K>2
    greater_is_better=True,
    stratified=True,
    score_uses_decision=False,          # balanced accuracy needs predicted labels
    behavior_regression=False,
    map_strategy="ovr",
    normalize_behavior=False,
    report_name="Support Vector Classification (SVC, one-vs-rest)",
    default_grid={"C": _C_GRID, "gamma": _GAMMA_GRID},
)
