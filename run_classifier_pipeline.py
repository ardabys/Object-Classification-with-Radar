"""
Complete classifier pipeline runner.

This script calls:
1. training_classifier.run_training_pipeline(...)
2. testing_classifier.run_testing_pipeline(...)

It connects the two stages by passing the selected features and best SVM
hyperparameters from training directly into testing. The same information is
also saved to trained_classifier_config.json, so testing_classifier.py can be
run separately if needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from training_classifier import run_training_pipeline
from testing_classifier import run_testing_pipeline

def run_full_pipeline(
    training_csv_path: str | Path = "data/training_features.csv",
    testing_csv_path: str | Path = "data/testing_features.csv",
    file_col: str = "File",
    label_col: str = "Activity",
    n_splits: int = 5,
    n_features_to_select: int = 5,
    auto_choose_k: bool = False,
    run_sfs_subset_test: bool = False,
    subset_results_csv: str | Path = "data/sfs_subset_size_results.csv",
    output_config_path: str | Path = "data/trained_classifier_config.json",
    test_confusion_matrix_path: str | Path = "figures/test_confusion_matrix.png",
    show_plots: bool = False,
) -> dict:
    """Run training and testing in one call."""
    print("\n==============================")
    print("RUNNING TRAINING PIPELINE")
    print("==============================")

    training_config = run_training_pipeline(
        training_csv_path=training_csv_path,
        file_col=file_col,
        label_col=label_col,
        n_splits=n_splits,
        run_sfs_subset_test=run_sfs_subset_test,
        subset_results_csv=subset_results_csv,
        n_features_to_select=n_features_to_select,
        auto_choose_k=auto_choose_k,
        output_config_path=output_config_path,
        show_plots=show_plots,
    )

    print("\n=============================")
    print("RUNNING TESTING PIPELINE")
    print("=============================")

    best_params = training_config["best_params"]
    testing_results = run_testing_pipeline(
        training_csv_path=training_csv_path,
        testing_csv_path=testing_csv_path,
        config_path=output_config_path,
        selected_features=training_config["selected_features"],
        best_C=best_params["C"],
        best_gamma=best_params["gamma"],
        file_col=file_col,
        label_col=label_col,
        confusion_matrix_save_path=test_confusion_matrix_path,
        show_plots=show_plots,
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

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full radar activity classification pipeline.")
    parser.add_argument("--training_csv", default="data/training_features.csv")
    parser.add_argument("--testing_csv", default="data/testing_features.csv")
    parser.add_argument("--file_col", default="File")
    parser.add_argument("--label_col", default="Activity")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_features", type=int, default=5)
    parser.add_argument("--auto_choose_k", action="store_true")
    parser.add_argument("--run_sfs_subset_test", action="store_true")
    parser.add_argument("--subset_results_csv", default="data/sfs_subset_size_results.csv")
    parser.add_argument("--output_config", default="data/trained_classifier_config.json")
    parser.add_argument("--test_confusion_matrix", default="figures/test_confusion_matrix.png")
    parser.add_argument("--show_plots", action="store_true")
    return parser

def main() -> dict:
    args = build_arg_parser().parse_args()
    return run_full_pipeline(
        training_csv_path=args.training_csv,
        testing_csv_path=args.testing_csv,
        file_col=args.file_col,
        label_col=args.label_col,
        n_splits=args.n_splits,
        n_features_to_select=args.n_features,
        auto_choose_k=args.auto_choose_k,
        run_sfs_subset_test=args.run_sfs_subset_test,
        subset_results_csv=args.subset_results_csv,
        output_config_path=args.output_config,
        test_confusion_matrix_path=args.test_confusion_matrix,
        show_plots=args.show_plots,
    )

if __name__ == "__main__":
    main()
