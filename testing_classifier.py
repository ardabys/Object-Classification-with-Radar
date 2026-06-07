import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. SETTINGS
training_csv_path = "training_features.csv"
testing_csv_path = "testing_features.csv"

file_col = "File"
label_col = "Activity"

# Fill these with the selected features from your training pipeline
selected_features = ['Maximum_Velocity_Bandwidth',
                     'Active_Motion_Fraction',
                     'Energy_Weighted_Velocity',
                     'Skewness_Doppler_Distr',
                     'Total_Signal_Over_Max']

# Fill these with the best parameters from GridSearchCV
best_C = 1
best_gamma = "scale"
confusion_matrix_save_path = "test_confusion_matrix.png"

# 2. DATA LOADING
def load_data(csv_path):
    """
    Load a feature CSV file.
    """
    df = pd.read_csv(csv_path)
    return df

def split_data(df, selected_features, label_col="Activity"):
    """
    Split dataframe into selected feature matrix X and labels y.
    """
    X = df[selected_features]
    y = df[label_col]
    return X, y

def check_selected_features_exist(df, selected_features):
    """
    Check that all selected features exist in the dataframe.
    """
    missing_features = [
        feature for feature in selected_features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "The following selected features are missing from the dataset:\n"
            + "\n".join(missing_features)
        )

# 3. MODEL DEFINITION
def create_final_model(C, gamma):
    """
    Create final tuned RBF-SVM model.

    StandardScaler is included because SVM depends on feature scale.
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            C=C,
            gamma=gamma
        ))
    ])
    return model

# 4. TEST EVALUATION
def evaluate_on_test_set(model, X_test, y_test):
    """
    Evaluate trained model on unseen test data.
    """
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


def plot_and_save_confusion_matrix(y_test, y_pred, labels, save_path):
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    disp.plot()

    plt.title("Confusion Matrix - Unseen Test Set")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"\nConfusion matrix saved to: {save_path}")

# 5. MAIN TESTING PIPELINE
def main():
    # Load testing data
    test_df = load_data(testing_csv_path)

    check_selected_features_exist(
        df=test_df,
        selected_features=selected_features
    )

    X_test, y_test = split_data(
        df=test_df,
        selected_features=selected_features,
        label_col=label_col
    )
    # Load training data
    train_df = load_data(training_csv_path)

    check_selected_features_exist(
        df=train_df,
        selected_features=selected_features
    )

    X_train, y_train = split_data(
        df=train_df,
        selected_features=selected_features,
        label_col=label_col
    )
    # Train final model on full training set
    final_model = create_final_model(
        C=best_C,
        gamma=best_gamma
    )

    final_model.fit(X_train, y_train)
    # Evaluate once on unseen test set

    y_pred, accuracy, macro_f1 = evaluate_on_test_set(
        model=final_model,
        X_test=X_test,
        y_test=y_test
    )

    # Confusion matrix
    labels = sorted(y_train.unique())

    plot_and_save_confusion_matrix(
        y_test=y_test,
        y_pred=y_pred,
        labels=labels,
        save_path=confusion_matrix_save_path
    )

    print("\nTesting pipeline completed.")
    print("---------------------------")
    print("Selected features used:")
    for feature in selected_features:
        print("-", feature)

    print("\nModel hyperparameters:")
    print("C:", best_C)
    print("gamma:", best_gamma)

if __name__ == "__main__":
    main()