import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import SequentialFeatureSelector

# 1. DATA LOADING AND SPLITTING
def load_training_data(csv_path):
    """
    Load the training feature CSV.
    """
    df = pd.read_csv(csv_path)
    return df

def split_features_labels_groups(df, file_col="File", label_col="Activity"):
    """
    Split dataframe into:
    - X: feature matrix
    - y: activity labels
    - groups: subject IDs extracted from filename
    - feature_cols: feature names
    """
    feature_cols = df.columns[2:].tolist()

    X = df[feature_cols]
    y = df[label_col]
    # Example filename: 1P36A01R01.dat
    # This extracts subject 36.
    groups = df[file_col].str.extract(r"P(\d+)")[0]
    return X, y, groups, feature_cols

def print_dataset_overview(df, X, y, groups, feature_cols):
    """
    Print basic dataset information.
    """
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


# 2. CROSS-VALIDATION SETUP
def create_group_cv_splits(X, y, groups, n_splits=5):
    """
    Create subject-wise CV splits.

    GroupKFold ensures that samples from the same subject are not
    present in both training and validation folds.
    """
    group_cv = GroupKFold(n_splits=n_splits)
    cv_splits = list(group_cv.split(X, y, groups=groups))

    return cv_splits

# 3. MODEL DEFINITION
def create_svm_model(C=1, gamma="scale"):
    """
    Create an RBF-SVM model with standardization.

    StandardScaler is inside the pipeline to avoid data leakage.
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", C=C, gamma=gamma))
    ])

    return model

# 4. GENERAL EVALUATION FUNCTIONS
def evaluate_cv_scores(model, X, y, cv_splits, scoring):
    """
    Evaluate model using cross-validation.
    """
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv_splits,
        scoring=scoring,
        n_jobs=-1
    )
    return scores

def print_cv_result(name, scores):
    """
    Print mean and standard deviation of CV scores.
    """
    print(f"{name}: {scores.mean():.3f} ± {scores.std():.3f}")

def evaluate_accuracy_and_macro_f1(model, X, y, cv_splits, name):
    """
    Evaluate model using accuracy and macro F1.
    """
    acc_scores = evaluate_cv_scores(
        model=model,
        X=X,
        y=y,
        cv_splits=cv_splits,
        scoring="accuracy"
    )

    f1_scores = evaluate_cv_scores(
        model=model,
        X=X,
        y=y,
        cv_splits=cv_splits,
        scoring="f1_macro"
    )

    print_cv_result(f"{name} Accuracy", acc_scores)
    print_cv_result(f"{name} Macro F1", f1_scores)

    return acc_scores, f1_scores

# 5. BASELINE MODEL WITH ALL FEATURES
def evaluate_baseline_all_features(X, y, cv_splits):
    """
    Evaluate default RBF-SVM using all features.
    """
    print("\nBaseline model using all features")
    print("---------------------------------")

    model = create_svm_model(C=1, gamma="scale")

    acc_scores, f1_scores = evaluate_accuracy_and_macro_f1(
        model=model,
        X=X,
        y=y,
        cv_splits=cv_splits,
        name="All features baseline"
    )

    return acc_scores, f1_scores

# 6. SINGLE-FEATURE EVALUATION
def evaluate_single_features(X, y, feature_cols, cv_splits):
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

        f1_scores = evaluate_cv_scores(
            model=model,
            X=Xi,
            y=y,
            cv_splits=cv_splits,
            scoring="f1_macro"
        )

        acc_scores = evaluate_cv_scores(
            model=model,
            X=Xi,
            y=y,
            cv_splits=cv_splits,
            scoring="accuracy"
        )

        results.append({
            "Feature": feature,
            "Macro_F1_mean": f1_scores.mean(),
            "Macro_F1_std": f1_scores.std(),
            "Accuracy_mean": acc_scores.mean(),
            "Accuracy_std": acc_scores.std()
        })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "Macro_F1_mean",
        ascending=False
    ).reset_index(drop=True)

    print(results_df)

    return results_df

# 7. SEQUENTIAL FORWARD SELECTION
def run_sfs(X, y, cv_splits, n_features_to_select):
    """
    Run Sequential Forward Selection with default RBF-SVM.
    """
    base_model = create_svm_model(C=1, gamma="scale")

    sfs = SequentialFeatureSelector(
        estimator=base_model,
        n_features_to_select=n_features_to_select,
        direction="forward",
        scoring="f1_macro",
        cv=cv_splits,
        n_jobs=-1
    )

    sfs.fit(X, y)

    selected_features = X.columns[sfs.get_support()].tolist()

    return selected_features, sfs


def test_increasing_number_of_features(X, y, feature_cols, cv_splits):
    """
    Test SFS for k = 1, 2, ..., number of features - 1.
    Then add the full feature set result separately.
    """
    print("\nTesting increasing number of selected features")
    print("----------------------------------------------")

    results = []

    for k in range(1, len(feature_cols)):
        selected_features, _ = run_sfs(
            X=X,
            y=y,
            cv_splits=cv_splits,
            n_features_to_select=k
        )

        model = create_svm_model(C=1, gamma="scale")

        f1_scores = evaluate_cv_scores(
            model=model,
            X=X[selected_features],
            y=y,
            cv_splits=cv_splits,
            scoring="f1_macro"
        )

        acc_scores = evaluate_cv_scores(
            model=model,
            X=X[selected_features],
            y=y,
            cv_splits=cv_splits,
            scoring="accuracy"
        )

        results.append({
            "Number_of_features": k,
            "Selected_features": selected_features,
            "Macro_F1_mean": f1_scores.mean(),
            "Macro_F1_std": f1_scores.std(),
            "Accuracy_mean": acc_scores.mean(),
            "Accuracy_std": acc_scores.std()
        })

        print(
            f"{k} features: "
            f"Macro F1 = {f1_scores.mean():.3f} ± {f1_scores.std():.3f}, "
            f"Accuracy = {acc_scores.mean():.3f} ± {acc_scores.std():.3f}"
        )
        print("Selected:", selected_features)
        print()

    # Full feature set, evaluated without SFS
    model = create_svm_model(C=1, gamma="scale")

    full_f1_scores = evaluate_cv_scores(
        model=model,
        X=X,
        y=y,
        cv_splits=cv_splits,
        scoring="f1_macro"
    )

    full_acc_scores = evaluate_cv_scores(
        model=model,
        X=X,
        y=y,
        cv_splits=cv_splits,
        scoring="accuracy"
    )

    results.append({
        "Number_of_features": len(feature_cols),
        "Selected_features": feature_cols,
        "Macro_F1_mean": full_f1_scores.mean(),
        "Macro_F1_std": full_f1_scores.std(),
        "Accuracy_mean": full_acc_scores.mean(),
        "Accuracy_std": full_acc_scores.std()
    })

    print(
        f"{len(feature_cols)} features, full feature set: "
        f"Macro F1 = {full_f1_scores.mean():.3f} ± {full_f1_scores.std():.3f}, "
        f"Accuracy = {full_acc_scores.mean():.3f} ± {full_acc_scores.std():.3f}"
    )

    results_df = pd.DataFrame(results)

    return results_df

def choose_number_of_features(subset_results_df):
    """
    Choose the number of features.
    Current rule:
    - choose the k with the highest mean Macro F1.
    """
    best_row = subset_results_df.loc[
        subset_results_df["Macro_F1_mean"].idxmax()
    ]

    best_k = int(best_row["Number_of_features"])

    print("\nChosen number of features")
    print("-------------------------")
    print("Best k:", best_k)
    print(f"Best Macro F1: {best_row['Macro_F1_mean']:.3f}")

    return best_k

def select_final_features(X, y, cv_splits, n_features_to_select):
    """
    Run final SFS with chosen number of features.
    """
    print("\nFinal sequential forward feature selection")
    print("------------------------------------------")

    selected_features, sfs = run_sfs(
        X=X,
        y=y,
        cv_splits=cv_splits,
        n_features_to_select=n_features_to_select
    )

    print("Selected features:")
    for feature in selected_features:
        print("-", feature)

    return selected_features, sfs


# 8. HYPERPARAMETER GRID SEARCH
def tune_svm_hyperparameters(X_selected, y, cv_splits):
    """
    Tune C and gamma for RBF-SVM using GridSearchCV.

    This is done after feature selection.
    """
    print("\nHyperparameter grid search")
    print("--------------------------")

    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf"))
    ])

    param_grid = {
        "classifier__C": [0.1, 1, 10, 100],
        "classifier__gamma": ["scale", 0.001, 0.01, 0.1, 1]
    }

    grid_search = GridSearchCV(
        estimator=svm_pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv_splits,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_selected, y)

    print("\nBest parameters:")
    print(grid_search.best_params_)

    print(f"Best CV Macro F1: {grid_search.best_score_:.3f}")

    return grid_search



# 9. FINAL TRAINING-CV EVALUATION
def evaluate_tuned_selected_model(grid_search, X_selected, y, cv_splits):
    """
    Evaluate the tuned model with selected features using CV.
    """
    print("\nSelected features + tuned SVM performance")
    print("-----------------------------------------")

    best_model = grid_search.best_estimator_

    acc_scores, f1_scores = evaluate_accuracy_and_macro_f1(
        model=best_model,
        X=X_selected,
        y=y,
        cv_splits=cv_splits,
        name="Selected + tuned SVM"
    )

    return acc_scores, f1_scores

def plot_cv_confusion_matrix(model, X, y, cv_splits, title, save_path = None):
    """
    Plot cross-validated confusion matrix.
    """
    y_pred = cross_val_predict(
        model,
        X,
        y,
        cv=cv_splits,
        n_jobs=-1
    )

    print("\nClassification report")
    print("---------------------")
    print(classification_report(y, y_pred))

    labels = sorted(y.unique())

    cm = confusion_matrix(
        y,
        y_pred,
        labels=labels
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    disp.plot()
    plt.title(title)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

def plot_performance_vs_number_of_features(
    subset_results_df,
    metric_col="Macro_F1_mean",
    std_col="Macro_F1_std",
    ylabel="Macro F1 (%)",
    title="Classification Performance with Increasing Number of Features",
    save_path=None
):
    """
    Plot model performance as a function of the number of selected features.
    """

    x = subset_results_df["Number_of_features"]
    y = subset_results_df[metric_col] * 100

    plt.figure(figsize=(8, 5))

    plt.plot(
        x,
        y,
        marker="o",
        linewidth=2
    )

    if std_col is not None and std_col in subset_results_df.columns:
        y_std = subset_results_df[std_col] * 100

        plt.fill_between(
            x,
            y - y_std,
            y + y_std,
            alpha=0.2
        )

    plt.xlabel("Number of Features")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.xticks(x)
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

def load_subset_size_results(csv_path):
    results_df = pd.read_csv(csv_path)
    return results_df

# 10. MAIN TRAINING PIPELINE
def main():
    training_csv_path = "training_features.csv"
    file_col = "File"
    label_col = "Activity"
    n_splits = 5 # CV splits.

    # If True, reruns SFS subset-size test, which takes long time.
    # If False, loads previous results from CSV.
    run_sfs_subset_test = False
    subset_results_csv = "sfs_subset_size_results.csv"

    # Load and prepare training data
    df = load_training_data(training_csv_path)

    X, y, groups, feature_cols = split_features_labels_groups(
        df=df,
        file_col=file_col,
        label_col=label_col
    )

    print_dataset_overview(
        df=df,
        X=X,
        y=y,
        groups=groups,
        feature_cols=feature_cols
    )

    # Create subject-wise CV splits
    cv_splits = create_group_cv_splits(
        X=X,
        y=y,
        groups=groups,
        n_splits=n_splits
    )

    # 1. Baseline performance with all features
    evaluate_baseline_all_features(
        X=X,
        y=y,
        cv_splits=cv_splits
    )

    # 2. Single-feature evaluation
    single_feature_results = evaluate_single_features(
        X=X,
        y=y,
        feature_cols=feature_cols,
        cv_splits=cv_splits
    )

    # 3. Test or load increasing number of selected features
    if run_sfs_subset_test:
        subset_results_df = test_increasing_number_of_features(
            X=X,
            y=y,
            feature_cols=feature_cols,
            cv_splits=cv_splits
        )

        subset_results_df.to_csv(subset_results_csv, index=False)

    else:
        subset_results_df = load_subset_size_results(
            csv_path=subset_results_csv
        )

    plot_performance_vs_number_of_features(
        subset_results_df=subset_results_df,
        metric_col="Macro_F1_mean",
        std_col="Macro_F1_std",
        ylabel="Macro F1 (%)",
        title="Macro F1 with Increasing Number of Features",
        save_path="macro_f1_vs_number_of_features.png"
    )

    plot_performance_vs_number_of_features(
        subset_results_df=subset_results_df,
        metric_col="Accuracy_mean",
        std_col="Accuracy_std",
        ylabel="Accuracy (%)",
        title="Accuracy with Increasing Number of Features",
        save_path="accuracy_vs_number_of_features.png"
    )

    # save results
    subset_results_df.to_csv("sfs_subset_size_results.csv", index=False)
    single_feature_results.to_csv("single_feature_results.csv", index=False)

    # 4. Choose number of features

    # chosen_k = choose_number_of_features(
    #     subset_results_df=subset_results_df
    # )

    # 5. Final feature selection with chosen k = 5
    selected_features, sfs = select_final_features(
        X=X,
        y=y,
        cv_splits=cv_splits,
        n_features_to_select=5
    )

    X_selected = X[selected_features]

    # 6. Hyperparameter tuning on selected features
    grid_search = tune_svm_hyperparameters(
        X_selected=X_selected,
        y=y,
        cv_splits=cv_splits
    )

    # 7. Evaluate selected + tuned model with CV
    evaluate_tuned_selected_model(
        grid_search=grid_search,
        X_selected=X_selected,
        y=y,
        cv_splits=cv_splits
    )

    # 8. Confusion matrix with selected features + tuned model
    best_model = grid_search.best_estimator_

    plot_cv_confusion_matrix(
        model=best_model,
        X=X_selected,
        y=y,
        cv_splits=cv_splits,
        title="Training Confusion Matrix - Selected Features + Tuned SVM",
        save_path="training_cv_confusion_matrix.png"
    )

    print("\nTraining pipeline completed.")
    print("----------------------------")
    print("Final selected features:")
    print(selected_features)
    print("\nBest hyperparameters:")
    print(grid_search.best_params_)


if __name__ == "__main__":
    main()