"""
Testing pipeline for radar activity classification.

This module can be run directly, or imported by run_classifier_pipeline.py.
It loads the final selected features and best SVM hyperparameters from the
training configuration JSON unless these values are passed directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 1. DATA LOADING
def load_data(csv_path: str | Path) -> pd.DataFrame:
    """Load a feature CSV file."""
    return pd.read_csv(csv_path)


def split_data(
    df: pd.DataFrame,
    selected_features: list[str],
    label_col: str = "Activity",
) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into selected feature matrix X and labels y."""
    X = df[selected_features]
    y = df[label_col]
    return X, y

def ensure_parent_dir(path: str | Path) -> None:
    """
    Create the parent directory of a file path if it does not exist.
    """
    path = Path(path)

    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)

def check_selected_features_exist(df: pd.DataFrame, selected_features: list[str]) -> None:
    """Check that all selected features exist in the dataframe."""
    missing_features = [feature for feature in selected_features if feature not in df.columns]

    if missing_features:
        raise ValueError(
            "The following selected features are missing from the dataset:\n"
            + "\n".join(missing_features)
        )

def load_training_config(config_path: str | Path) -> dict[str, Any]:
    """Load selected features and best hyperparameters saved by training."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Could not find training configuration: {config_path}. "
            "Run training_classifier.py first or use run_classifier_pipeline.py."
        )

    return json.loads(config_path.read_text(encoding="utf-8"))


# 2. MODEL DEFINITION
def create_final_model(C: float, gamma: str | float) -> Pipeline:
    """
    Create final tuned RBF-SVM model.

    StandardScaler is included because SVM depends on feature scale.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", C=C, gamma=gamma)),
    ])


# 3. TEST EVALUATION
def evaluate_on_test_set(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.Series, float, float]:
    """Evaluate trained model on unseen test data."""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("\nFinal unseen test set performance")
    print("---------------------------------")
    print(f"Test Accuracy: {accuracy:.3f}")
    print(f"Test Macro F1: {macro_f1:.3f}")

    print("\nClassification report")
    print("---------------------")
    print(classification_report(y_test, y_pred))

    return y_pred, accuracy, macro_f1


def plot_and_save_confusion_matrix(
    y_test: pd.Series,
    y_pred: pd.Series,
    labels: list[Any],
    save_path: str | Path,
    show_plot: bool = False,
) -> None:
    """Plot and save confusion matrix for the unseen test set."""
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    disp.plot()
    plt.title("Confusion Matrix - Unseen Test Set")
    plt.tight_layout()
    ensure_parent_dir(save_path)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nConfusion matrix saved to: {save_path}")

    if show_plot:
        plt.show()
    plt.close()


# 4. MAIN TESTING PIPELINE
def run_testing_pipeline(
    training_csv_path: str | Path = "data/training_features.csv",
    testing_csv_path: str | Path = "data/testing_features.csv",
    config_path: str | Path = "data/trained_classifier_config.json",
    selected_features: list[str] | None = None,
    best_C: float | None = None,
    best_gamma: str | float | None = None,
    file_col: str = "File",
    label_col: str = "Activity",
    confusion_matrix_save_path: str | Path = "test_confusion_matrix.png",
    show_plots: bool = False,
) -> dict[str, Any]:
    """
    Run final training on the full training set and evaluate on test data.

    If selected_features, best_C, or best_gamma are not provided, they are
    loaded from config_path.
    """
    if selected_features is None or best_C is None or best_gamma is None:
        config = load_training_config(config_path)
        selected_features = selected_features or config["selected_features"]
        best_params = config["best_params"]
        best_C = best_C if best_C is not None else best_params["C"]
        best_gamma = best_gamma if best_gamma is not None else best_params["gamma"]

    # Load data.
    train_df = load_data(training_csv_path)
    test_df = load_data(testing_csv_path)

    for df_name, df in [("training", train_df), ("testing", test_df)]:
        if label_col not in df.columns:
            raise ValueError(f"The {df_name} dataframe is missing label column: {label_col}")
        if file_col not in df.columns:
            raise ValueError(f"The {df_name} dataframe is missing file column: {file_col}")
        check_selected_features_exist(df, selected_features)

    X_train, y_train = split_data(train_df, selected_features, label_col=label_col)
    X_test, y_test = split_data(test_df, selected_features, label_col=label_col)

    # Train final model on full training set.
    final_model = create_final_model(C=best_C, gamma=best_gamma)
    final_model.fit(X_train, y_train)

    # Evaluate once on unseen test set.
    y_pred, accuracy, macro_f1 = evaluate_on_test_set(final_model, X_test, y_test)

    labels = sorted(y_train.unique())
    plot_and_save_confusion_matrix(
        y_test=y_test,
        y_pred=y_pred,
        labels=labels,
        save_path=confusion_matrix_save_path,
        show_plot=show_plots,
    )

    print("\nTesting pipeline completed.")
    print("---------------------------")
    print("Selected features used:")
    for feature in selected_features:
        print("-", feature)

    print("\nModel hyperparameters:")
    print("C:", best_C)
    print("gamma:", best_gamma)

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "selected_features": selected_features,
        "best_params": {"C": best_C, "gamma": best_gamma},
        "confusion_matrix_save_path": str(confusion_matrix_save_path),
    }



# SCRIPT SETTINGS
if __name__ == "__main__":
    run_testing_pipeline(
        training_csv_path="data/training_features.csv",
        testing_csv_path="data/testing_features.csv",
        config_path="data/trained_classifier_config.json",
        selected_features=None,
        best_C=None,
        best_gamma=None,
        file_col="File",
        label_col="Activity",
        confusion_matrix_save_path="figures/test_confusion_matrix.png",
        show_plots=False,
    )
