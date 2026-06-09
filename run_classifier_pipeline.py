"""
Complete classifier pipeline runner.

This script runs:
1. training_classifier.run_training_pipeline(...)
2. testing_classifier.run_testing_pipeline(...)

It passes the selected features and best SVM hyperparameters from training
directly into testing.
"""

from pathlib import Path
from training_classifier import run_training_pipeline
from testing_classifier import run_testing_pipeline

# SETTINGS
TRAINING_CSV_PATH = Path("data/training_features.csv")
TESTING_CSV_PATH = Path("data/testing_features.csv")

FILE_COL = "File"
LABEL_COL = "Activity"

N_SPLITS = 5
N_FEATURES_TO_SELECT = 5
AUTO_CHOOSE_K = False

# If True, recompute the expensive SFS subset-size test.
# If False, the code should load the existing CSV or create it if missing.
RUN_SFS_SUBSET_TEST = False

SUBSET_RESULTS_CSV = Path("data/sfs_subset_size_results.csv")
SINGLE_FEATURE_RESULTS_CSV = Path("data/single_feature_results.csv")
OUTPUT_CONFIG_PATH = Path("data/trained_classifier_config.json")

TEST_CONFUSION_MATRIX_PATH = Path("figures/test_confusion_matrix.png")

SHOW_PLOTS = False


# FULL PIPELINE
def run_full_pipeline() -> dict:
    """Run training and testing in one call."""
    print("\n==============================")
    print("RUNNING TRAINING PIPELINE")
    print("==============================")

    training_config = run_training_pipeline(
        training_csv_path=TRAINING_CSV_PATH,
        file_col=FILE_COL,
        label_col=LABEL_COL,
        n_splits=N_SPLITS,
        run_sfs_subset_test=RUN_SFS_SUBSET_TEST,
        subset_results_csv=SUBSET_RESULTS_CSV,
        single_feature_results_csv=SINGLE_FEATURE_RESULTS_CSV,
        n_features_to_select=N_FEATURES_TO_SELECT,
        auto_choose_k=AUTO_CHOOSE_K,
        output_config_path=OUTPUT_CONFIG_PATH,
        show_plots=SHOW_PLOTS,
    )

    print("\n=============================")
    print("RUNNING TESTING PIPELINE")
    print("=============================")

    best_params = training_config["best_params"]

    testing_results = run_testing_pipeline(
        training_csv_path=TRAINING_CSV_PATH,
        testing_csv_path=TESTING_CSV_PATH,
        config_path=OUTPUT_CONFIG_PATH,
        selected_features=training_config["selected_features"],
        best_C=best_params["C"],
        best_gamma=best_params["gamma"],
        file_col=FILE_COL,
        label_col=LABEL_COL,
        confusion_matrix_save_path=TEST_CONFUSION_MATRIX_PATH,
        show_plots=SHOW_PLOTS,
    )

    print("\n=====================")
    print("FULL PIPELINE FINISHED")
    print("=====================")
    print("Selected features:", training_config["selected_features"])
    print("Best parameters:", training_config["best_params"])
    print(f"Test Accuracy: {testing_results['accuracy']:.3f}")
    print(f"Test Macro F1: {testing_results['macro_f1']:.3f}")

    return {
        "training_config": training_config,
        "testing_results": testing_results,
    }


if __name__ == "__main__":
    run_full_pipeline()