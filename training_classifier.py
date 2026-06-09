"""
Training pipeline for radar activity classification.

This module can be run directly, or imported by run_classifier_pipeline.py.
It performs:
1. Subject-wise cross-validation using GroupKFold.
2. Baseline evaluation with all features.
3. Single-feature evaluation.
4. Sequential forward selection.
5. RBF-SVM hyperparameter tuning.
6. Cross-validated evaluation and confusion matrix plotting.
7. Saving the selected features and best hyperparameters to JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, GridSearchCV, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 1. DATA LOADING AND SPLITTING
def load_training_data(csv_path: str | Path) -> pd.DataFrame:
    """Load the training feature CSV."""
    return pd.read_csv(csv_path)

def split_features_labels_groups(
    df: pd.DataFrame,
    file_col: str = "File",
    label_col: str = "Activity",
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """
    Split dataframe into:
    - X: feature matrix
    - y: activity labels
    - groups: subject IDs extracted from filename
    - feature_cols: feature names
    """
    required_cols = {file_col, label_col}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    feature_cols = [col for col in df.columns if col not in [file_col, label_col]]
    if not feature_cols:
        raise ValueError("No feature columns found.")

    X = df[feature_cols]
    y = df[label_col]

    # Example filename: 1P36A01R01.dat -> subject group 36.
    groups = df[file_col].astype(str).str.extract(r"P(\d+)")[0]
    if groups.isna().any():
        bad_files = df.loc[groups.isna(), file_col].astype(str).tolist()[:5]
        raise ValueError(
            "Could not extract subject IDs from some filenames. "
            f"Examples: {bad_files}"
        )
    return X, y, groups, feature_cols


def print_dataset_overview(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    feature_cols: list[str],
) -> None:
    """Print basic dataset information."""
    print("\nDataset overview")
    print("----------------")
    print("Samples:", len(df))
    print("Features:", len(feature_cols))
    print("Subjects:", groups.nunique())

    print("\nClass balance:")
    print(y.value_counts().sort_index())

    print("\nFeature columns:")
    for feature in feature_cols:
        print("-", feature)

def ensure_parent_dir(path: str | Path) -> None:
    """
    Create the parent directory of a file path if it does not exist.
    """
    path = Path(path)

    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)


# 2. CROSS-VALIDATION SETUP
def create_group_cv_splits(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
) -> list[tuple[Any, Any]]:
    """
    Create subject-wise CV splits.

    GroupKFold ensures that samples from the same subject are not present in
    both training and validation folds.
    """
    n_groups = groups.nunique()
    if n_splits > n_groups:
        raise ValueError(
            f"n_splits={n_splits} is larger than the number of groups={n_groups}."
        )

    group_cv = GroupKFold(n_splits=n_splits)
    return list(group_cv.split(X, y, groups=groups))


# 3. MODEL DEFINITION
def create_svm_model(C: float = 1, gamma: str | float = "scale") -> Pipeline:
    """
    Create an RBF-SVM model with standardization.

    StandardScaler is inside the pipeline to avoid data leakage.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", C=C, gamma=gamma)),
    ])


# 4. GENERAL EVALUATION FUNCTIONS
def evaluate_cv_scores(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: list[tuple[Any, Any]],
    scoring: str,
) -> Any:
    """Evaluate model using cross-validation."""
    return cross_val_score(
        model,
        X,
        y,
        cv=cv_splits,
        scoring=scoring,
        n_jobs=-1,
    )


def print_cv_result(name: str, scores: Any) -> None:
    """Print mean and standard deviation of CV scores."""
    print(f"{name}: {scores.mean():.3f} ± {scores.std():.3f}")

def evaluate_accuracy_and_macro_f1(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: list[tuple[Any, Any]],
    name: str,
) -> tuple[Any, Any]:
    """Evaluate model using accuracy and macro F1."""
    acc_scores = evaluate_cv_scores(model, X, y, cv_splits, scoring="accuracy")
    f1_scores = evaluate_cv_scores(model, X, y, cv_splits, scoring="f1_macro")

    print_cv_result(f"{name} Accuracy", acc_scores)
    print_cv_result(f"{name} Macro F1", f1_scores)

    return acc_scores, f1_scores


# 5. BASELINE MODEL WITH ALL FEATURES
def evaluate_baseline_all_features(
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: list[tuple[Any, Any]],
) -> tuple[Any, Any]:
    """Evaluate default RBF-SVM using all features."""
    print("\nBaseline model using all features")
    print("---------------------------------")

    model = create_svm_model(C=1, gamma="scale")
    return evaluate_accuracy_and_macro_f1(
        model=model,
        X=X,
        y=y,
        cv_splits=cv_splits,
        name="All features baseline",
    )


# 6. SINGLE-FEATURE EVALUATION
def evaluate_single_features(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    cv_splits: list[tuple[Any, Any]],
) -> pd.DataFrame:
    """
    Evaluate each feature individually using the default RBF-SVM.

    Saves both Macro F1 and Accuracy for each single feature.
    """
    print("\nSingle-feature evaluation")
    print("-------------------------")

    results = []

    for feature in feature_cols:
        Xi = X[[feature]]
        model = create_svm_model(C=1, gamma="scale")

        f1_scores = evaluate_cv_scores(model, Xi, y, cv_splits, scoring="f1_macro")
        acc_scores = evaluate_cv_scores(model, Xi, y, cv_splits, scoring="accuracy")

        results.append({
            "Feature": feature,
            "Macro_F1_mean": f1_scores.mean(),
            "Macro_F1_std": f1_scores.std(),
            "Accuracy_mean": acc_scores.mean(),
            "Accuracy_std": acc_scores.std(),
        })

    results_df = pd.DataFrame(results).sort_values(
        "Macro_F1_mean",
        ascending=False,
    ).reset_index(drop=True)

    print(results_df)
    return results_df


# 7. SEQUENTIAL FORWARD SELECTION
def run_sfs(
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: list[tuple[Any, Any]],
    n_features_to_select: int,
) -> tuple[list[str], SequentialFeatureSelector]:
    """Run Sequential Forward Selection with default RBF-SVM."""
    base_model = create_svm_model(C=1, gamma="scale")

    sfs = SequentialFeatureSelector(
        estimator=base_model,
        n_features_to_select=n_features_to_select,
        direction="forward",
        scoring="f1_macro",
        cv=cv_splits,
        n_jobs=-1,
    )

    sfs.fit(X, y)
    selected_features = X.columns[sfs.get_support()].tolist()

    return selected_features, sfs


def test_increasing_number_of_features(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    cv_splits: list[tuple[Any, Any]],
) -> pd.DataFrame:
    """
    Test SFS for increasing number of features.
    """
    print("\nTesting increasing number of selected features")
    print("----------------------------------------------")

    results = []

    for k in range(1, len(feature_cols)):
        selected_features, _ = run_sfs(X, y, cv_splits, n_features_to_select=k)
        model = create_svm_model(C=1, gamma="scale")

        f1_scores = evaluate_cv_scores(
            model,
            X[selected_features],
            y,
            cv_splits,
            scoring="f1_macro",
        )
        acc_scores = evaluate_cv_scores(
            model,
            X[selected_features],
            y,
            cv_splits,
            scoring="accuracy",
        )

        results.append({
            "Number_of_features": k,
            "Selected_features": selected_features,
            "Macro_F1_mean": f1_scores.mean(),
            "Macro_F1_std": f1_scores.std(),
            "Accuracy_mean": acc_scores.mean(),
            "Accuracy_std": acc_scores.std(),
        })

        print(
            f"{k} features: "
            f"Macro F1 = {f1_scores.mean():.3f} ± {f1_scores.std():.3f}, "
            f"Accuracy = {acc_scores.mean():.3f} ± {acc_scores.std():.3f}"
        )
        print("Selected:", selected_features)
        print()

    # Full feature set, evaluated without SFS.
    model = create_svm_model(C=1, gamma="scale")
    full_f1_scores = evaluate_cv_scores(model, X, y, cv_splits, scoring="f1_macro")
    full_acc_scores = evaluate_cv_scores(model, X, y, cv_splits, scoring="accuracy")

    results.append({
        "Number_of_features": len(feature_cols),
        "Selected_features": feature_cols,
        "Macro_F1_mean": full_f1_scores.mean(),
        "Macro_F1_std": full_f1_scores.std(),
        "Accuracy_mean": full_acc_scores.mean(),
        "Accuracy_std": full_acc_scores.std(),
    })

    print(
        f"{len(feature_cols)} features, full feature set: "
        f"Macro F1 = {full_f1_scores.mean():.3f} ± {full_f1_scores.std():.3f}, "
        f"Accuracy = {full_acc_scores.mean():.3f} ± {full_acc_scores.std():.3f}"
    )

    return pd.DataFrame(results)


def load_subset_size_results(csv_path: str | Path) -> pd.DataFrame:
    """Load previously saved SFS subset-size results."""
    return pd.read_csv(csv_path)


def choose_number_of_features(subset_results_df: pd.DataFrame) -> int:
    """
    Choose the number of features by the highest mean Macro F1.
    """
    best_row = subset_results_df.loc[subset_results_df["Macro_F1_mean"].idxmax()]
    best_k = int(best_row["Number_of_features"])

    print("\nChosen number of features")
    print("-------------------------")
    print("Best k:", best_k)
    print(f"Best Macro F1: {best_row['Macro_F1_mean']:.3f}")

    return best_k


def select_final_features(
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: list[tuple[Any, Any]],
    n_features_to_select: int,
) -> tuple[list[str], SequentialFeatureSelector]:
    """Run final SFS with the chosen number of features."""
    print("\nFinal sequential forward feature selection")
    print("------------------------------------------")

    selected_features, sfs = run_sfs(
        X=X,
        y=y,
        cv_splits=cv_splits,
        n_features_to_select=n_features_to_select,
    )

    print("Selected features:")
    for feature in selected_features:
        print("-", feature)

    return selected_features, sfs


# 8. HYPERPARAMETER GRID SEARCH
def tune_svm_hyperparameters(
    X_selected: pd.DataFrame,
    y: pd.Series,
    cv_splits: list[tuple[Any, Any]],
) -> GridSearchCV:
    """
    Tune C and gamma for RBF-SVM using GridSearchCV.

    This is done after feature selection.
    """
    print("\nHyperparameter grid search")
    print("--------------------------")

    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf")),
    ])

    param_grid = {
        "classifier__C": [0.1, 1, 10, 100],
        "classifier__gamma": ["scale", 0.001, 0.01, 0.1, 1],
    }

    grid_search = GridSearchCV(
        estimator=svm_pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv_splits,
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_selected, y)

    print("\nBest parameters:")
    print(grid_search.best_params_)
    print(f"Best CV Macro F1: {grid_search.best_score_:.3f}")

    return grid_search


# 9. FINAL TRAINING-CV EVALUATION AND PLOTS
def evaluate_tuned_selected_model(
    grid_search: GridSearchCV,
    X_selected: pd.DataFrame,
    y: pd.Series,
    cv_splits: list[tuple[Any, Any]],
) -> tuple[Any, Any]:
    """Evaluate the tuned model with selected features using CV."""
    print("\nSelected features + tuned SVM performance")
    print("-----------------------------------------")

    best_model = grid_search.best_estimator_
    return evaluate_accuracy_and_macro_f1(
        model=best_model,
        X=X_selected,
        y=y,
        cv_splits=cv_splits,
        name="Selected + tuned SVM",
    )


def plot_cv_confusion_matrix(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: list[tuple[Any, Any]],
    title: str,
    save_path: str | Path | None = None,
    show_plot: bool = False,
) -> None:
    """Plot cross-validated confusion matrix."""
    y_pred = cross_val_predict(model, X, y, cv=cv_splits, n_jobs=-1)

    print("\nClassification report")
    print("---------------------")
    print(classification_report(y, y_pred))

    labels = sorted(y.unique())
    cm = confusion_matrix(y, y_pred, labels=labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.title(title)
    plt.tight_layout()

    if save_path is not None:
        ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to: {save_path}")

    if show_plot:
        plt.show()
    plt.close()


def plot_performance_vs_number_of_features(
    subset_results_df: pd.DataFrame,
    metric_col: str = "Macro_F1_mean",
    std_col: str = "Macro_F1_std",
    ylabel: str = "Macro F1 (%)",
    title: str = "Classification Performance with Increasing Number of Features",
    save_path: str | Path | None = None,
    show_plot: bool = False,
) -> None:
    """Plot model performance as a function of selected feature count."""
    x = subset_results_df["Number_of_features"]
    y = subset_results_df[metric_col] * 100

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o", linewidth=2)

    if std_col is not None and std_col in subset_results_df.columns:
        y_std = subset_results_df[std_col] * 100
        plt.fill_between(x, y - y_std, y + y_std, alpha=0.2)

    plt.xlabel("Number of Features")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    if save_path is not None:
        ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Performance plot saved to: {save_path}")

    if show_plot:
        plt.show()
    plt.close()


def _json_safe(value: Any) -> Any:
    """Convert numpy/pandas values into JSON-safe Python values."""
    if hasattr(value, "item"):
        return value.item()
    return value


def save_training_config(
    config_path: str | Path,
    selected_features: list[str],
    grid_search: GridSearchCV,
    training_csv_path: str | Path,
    n_features_to_select: int,
) -> dict[str, Any]:
    """Save selected features and best hyperparameters for testing."""
    best_params = grid_search.best_params_

    config = {
        "training_csv_path": str(training_csv_path),
        "selected_features": selected_features,
        "n_features_to_select": int(n_features_to_select),
        "best_params": {
            "C": _json_safe(best_params["classifier__C"]),
            "gamma": _json_safe(best_params["classifier__gamma"]),
        },
        "best_cv_macro_f1": float(grid_search.best_score_),
    }

    config_path = Path(config_path)
    ensure_parent_dir(config_path)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"\nTraining configuration saved to: {config_path}")

    return config


# 10. MAIN TRAINING PIPELINE
def run_training_pipeline(
    training_csv_path: str | Path = "data/training_features.csv",
    file_col: str = "File",
    label_col: str = "Activity",
    n_splits: int = 5,
    run_sfs_subset_test: bool = True,
    subset_results_csv: str | Path = "data/sfs_subset_size_results.csv",
    single_feature_results_csv: str | Path = "data/single_feature_results.csv",
    n_features_to_select: int | None = 5,
    auto_choose_k: bool = False,
    output_config_path: str | Path = "data/trained_classifier_config.json",
    show_plots: bool = False,
) -> dict[str, Any]:
    """
    Run the complete training pipeline and return the final configuration.

    The returned dictionary is intended to be passed to the testing pipeline.
    """
    training_csv_path = Path(training_csv_path)

    # Load and prepare training data.
    df = load_training_data(training_csv_path)
    X, y, groups, feature_cols = split_features_labels_groups(df, file_col, label_col)

    print_dataset_overview(df, X, y, groups, feature_cols)

    # Create subject-wise CV splits.
    cv_splits = create_group_cv_splits(X, y, groups, n_splits=n_splits)

    # 1. Baseline performance with all features.
    evaluate_baseline_all_features(X, y, cv_splits)

    # 2. Single-feature evaluation.
    single_feature_results = evaluate_single_features(X, y, feature_cols, cv_splits)

    # 3. Test or load increasing number of selected features.
    subset_results_csv = Path(subset_results_csv)
    single_feature_results_csv = Path(single_feature_results_csv)

    if run_sfs_subset_test or not subset_results_csv.exists():
        if not subset_results_csv.exists():
            print(f"\nSubset results file not found: {subset_results_csv}")
            print("Running SFS subset-size test to create it.")

        subset_results_df = test_increasing_number_of_features(
            X, y, feature_cols, cv_splits
        )

        subset_results_csv.parent.mkdir(parents=True, exist_ok=True)
        subset_results_df.to_csv(subset_results_csv, index=False)

    else:
        subset_results_df = load_subset_size_results(subset_results_csv)

    plot_performance_vs_number_of_features(
        subset_results_df=subset_results_df,
        metric_col="Macro_F1_mean",
        std_col="Macro_F1_std",
        ylabel="Macro F1 (%)",
        title="Macro F1 with Increasing Number of Features",
        save_path="figures/macro_f1_vs_number_of_features.png",
        show_plot=show_plots,
    )

    plot_performance_vs_number_of_features(
        subset_results_df=subset_results_df,
        metric_col="Accuracy_mean",
        std_col="Accuracy_std",
        ylabel="Accuracy (%)",
        title="Accuracy with Increasing Number of Features",
        save_path="figures/accuracy_vs_number_of_features.png",
        show_plot=show_plots,
    )

    # Save intermediate results.
    subset_results_csv.parent.mkdir(parents=True, exist_ok=True)
    subset_results_df.to_csv(subset_results_csv, index=False)

    single_feature_results_csv.parent.mkdir(parents=True, exist_ok=True)
    single_feature_results.to_csv(single_feature_results_csv, index=False)

    # 4. Choose number of features.
    if auto_choose_k:
        n_features_to_select = choose_number_of_features(subset_results_df)
    elif n_features_to_select is None:
        raise ValueError("n_features_to_select must be set unless auto_choose_k=True.")

    if n_features_to_select < 1 or n_features_to_select > len(feature_cols):
        raise ValueError(
            "n_features_to_select must be between 1 and the number of features "
            f"({len(feature_cols)})."
        )

    # 5. Final feature selection.
    selected_features, _ = select_final_features(
        X=X,
        y=y,
        cv_splits=cv_splits,
        n_features_to_select=n_features_to_select,
    )
    selected_features = ['Maximum_Velocity_Bandwidth', 'Active_Motion_Fraction', 'Energy_Weighted_Velocity', 'Skewness_Doppler_Distr', 'Total_Signal_Over_Max']
    X_selected = X[selected_features]

    # 6. Hyperparameter tuning on selected features.
    grid_search = tune_svm_hyperparameters(X_selected, y, cv_splits)

    # 7. Evaluate selected + tuned model with CV.
    evaluate_tuned_selected_model(grid_search, X_selected, y, cv_splits)

    # 8. Confusion matrix with selected features + tuned model.
    best_model = grid_search.best_estimator_
    plot_cv_confusion_matrix(
        model=best_model,
        X=X_selected,
        y=y,
        cv_splits=cv_splits,
        title="Training Confusion Matrix - Selected Features + Tuned SVM",
        save_path="figures/training_cv_confusion_matrix.png",
        show_plot=show_plots,
    )

    config = save_training_config(
        config_path=output_config_path,
        selected_features=selected_features,
        grid_search=grid_search,
        training_csv_path=training_csv_path,
        n_features_to_select=n_features_to_select,
    )

    print("\nTraining pipeline completed.")
    print("----------------------------")
    print("Final selected features:")
    print(selected_features)
    print("\nBest hyperparameters:")
    print(config["best_params"])

    return config


# SCRIPT SETTINGS
if __name__ == "__main__":
    run_training_pipeline(
        training_csv_path="data/training_features.csv",
        file_col="File",
        label_col="Activity",
        n_splits=5,
        run_sfs_subset_test=False,
        subset_results_csv="data/sfs_subset_size_results.csv",
        single_feature_results_csv="data/single_feature_results.csv",
        n_features_to_select=5,
        auto_choose_k=False,
        output_config_path="data/trained_classifier_config.json",
        show_plots=False,
    )