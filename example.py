from svmLSMpy import run_svm_lsm

# Auto-routes to SVR (continuous behaviour) or SVC (categorical) based on the
# 'behavior' column. max_score is only used by the SVR path.
run_svm_lsm(
    symptom_folder="proj/data-final",
    csv_path="proj/data.csv",
    output_path="test-package-output",
    mode="auto",
    max_score=100,
)
