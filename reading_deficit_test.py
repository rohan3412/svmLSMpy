from svmLSMpy import run_svm_lsm

# Auto-routes to SVR (continuous behaviour) or SVC (categorical) based on the
# 'behavior' column. max_score is only used by the SVR path.
run_svm_lsm(
    symptom_folder="Z:\Research\Reading deficits\SCCAN Analysis\data",
    csv_path="Z:\Research\Reading deficits\SCCAN Analysis\Reading_L.csv",
    output_path="Z:\Research\Reading deficits\svmLSMpy_output/Lexical",
    mode="auto",
    max_score=1,
    n_permutations=100
)

run_svm_lsm(
    symptom_folder="Z:\Research\Reading deficits\SCCAN Analysis\data",
    csv_path="Z:\Research\Reading deficits\SCCAN Analysis\Reading_P.csv",
    output_path="Z:\Research\Reading deficits\svmLSMpy_output/Phonology",
    mode="auto",
    max_score=1,
    n_permutations=100
)
