"""
Stage 5 gate: auto-routing + public API.

Fast checks on _resolve_task (no pipeline run):
  - auto picks SVR / binary-SVC / multiclass-SVC from the behaviour column
  - mode='svr' / mode='svc' override the detection
  - run_svm_lsm requires max_score for the SVR path
Plus one end-to-end auto smoke to confirm run_svm_lsm(mode='auto') dispatches.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from svmLSMpy import run_svm_lsm
from svmLSMpy.run_svm_lsm import _resolve_task
from svmLSMpy.task import SVR_TASK, SVC_BINARY_TASK, SVC_MULTICLASS_TASK

DATA = Path(__file__).parent / "fixtures" / "data"
OUT = Path(__file__).parent / "_out" / "verify_routing"


def csv(name):
    return str(DATA / name / "behavior.csv")


def main():
    # auto-detection
    assert _resolve_task("auto", csv("svr")) is SVR_TASK
    assert _resolve_task("auto", csv("binary")) is SVC_BINARY_TASK
    assert _resolve_task("auto", csv("threeclass")) is SVC_MULTICLASS_TASK
    print("[1] auto picks SVR / binary-SVC / multiclass-SVC  OK")

    # overrides
    assert _resolve_task("svr", csv("binary")) is SVR_TASK          # force continuous
    assert _resolve_task("svc", csv("threeclass")) is SVC_MULTICLASS_TASK
    # svr fixture has 30 distinct continuous values -> forced svc = multiclass OvR
    assert _resolve_task("svc", csv("svr")) is SVC_MULTICLASS_TASK
    print("[2] mode='svr'/'svc' overrides win  OK")

    # max_score required for SVR
    try:
        run_svm_lsm(str(DATA / "svr" / "lesions"), csv("svr"), str(OUT), mode="svr")
        raise SystemExit("expected ValueError for missing max_score")
    except ValueError as e:
        assert "max_score" in str(e)
    print("[3] SVR path requires max_score  OK")

    # end-to-end auto smoke -> binary fixture should route to SVC and run
    OUT.mkdir(parents=True, exist_ok=True)
    results = run_svm_lsm(str(DATA / "binary" / "lesions"), csv("binary"), str(OUT),
                          mode="auto", behaviour_name="verify_auto",
                          param_grid={"C": [1], "gamma": ["scale"]},
                          n_permutations=10, n_splits=5, num_slices=3)
    folder = max(OUT.glob("*_results_*"), key=lambda p: p.stat().st_mtime)
    assert len(results) == 1 and (folder / "svc_lsm_report.html").exists()
    print("[4] run_svm_lsm(mode='auto') dispatched SVC end-to-end  OK")
    print("\nALL ROUTING CHECKS PASSED")


if __name__ == "__main__":
    main()
