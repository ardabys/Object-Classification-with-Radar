# Radar-Based Human Activity Classification

This repository contains a Python pipeline for classifying human activities from radar-derived feature tables. The classifier uses numerical features extracted from radar spectrograms and predicts one of six activities:

| Label | Activity |
|---|---|
| 1 | Walking back and forth |
| 2 | Sitting down on a chair |
| 3 | Standing up |
| 4 | Bending to pick up an object |
| 5 | Drinking from a cup |
| 6 | Falling down |

The pipeline trains an RBF-kernel Support Vector Machine (SVM), performs feature selection and hyperparameter tuning on the training data, and evaluates the final model on an unseen test set.

---

## Repository Structure

```text
project/
├── run_classifier_pipeline.py      # Runs training and testing in one go
├── training_classifier.py          # Training, feature selection, tuning, CV evaluation
├── testing_classifier.py           # Final evaluation on unseen test data
│
├── data/
│   ├── training_features.csv       # Training feature table
│   ├── testing_features.csv        # Unseen testing feature table
│   ├── sfs_subset_size_results.csv # Generated SFS subset-size results
│   ├── single_feature_results.csv  # Generated single-feature results
│   └── trained_classifier_config.json # Generated selected features + best parameters
│
└── figures/
    ├── macro_f1_vs_number_of_features.png
    ├── accuracy_vs_number_of_features.png
    ├── training_cv_confusion_matrix.png
    └── test_confusion_matrix.png
```

The `data/` and `figures/` folders are used for input data, intermediate results, configuration files, and saved plots.

---

## Input Data Format

Both `training_features.csv` and `testing_features.csv` should have the following structure:

```text
File, Activity, Feature_1, Feature_2, Feature_3, ...
```

The required columns are:

| Column | Description |
|---|---|
| `File` | Radar recording filename |
| `Activity` | Activity label, from 1 to 6 |
| Remaining columns | Numerical features used for classification |

The subject ID is extracted from the filename using the pattern `P<number>`. For example, `1P36A01R01.dat` is interpreted as subject `36`. This is used for subject-wise cross-validation.

---

## Installation

Create and activate a Python environment, then install the required packages:

```bash
pip install pandas numpy matplotlib scikit-learn
```

---

## Quick Start

Run the full training and testing pipeline with:

```bash
python run_classifier_pipeline.py
```

By default, this expects:

```text
data/training_features.csv
data/testing_features.csv
```

The script will:

1. Train the classifier on the training feature table.
2. Evaluate baseline and single-feature performance.
3. Select features using Sequential Forward Selection.
4. Tune the SVM hyperparameters using grid search.
5. Save the selected features and best parameters.
6. Train the final model on the full training set.
7. Evaluate the final model on the unseen test set.
8. Save the test confusion matrix.

---

## Main Commands

### Run the full pipeline

```bash
python run_classifier_pipeline.py
```

### Use custom data paths

```bash
python run_classifier_pipeline.py --training_csv path/to/training_features.csv --testing_csv path/to/testing_features.csv
```

### Select a specific number of features

```bash
python run_classifier_pipeline.py --n_features 5
```

### Recompute the expensive SFS subset-size test

```bash
python run_classifier_pipeline.py --run_sfs_subset_test
```

This evaluates model performance for increasing numbers of selected features and saves the results to:

```text
data/sfs_subset_size_results.csv
```

For later runs, omit `--run_sfs_subset_test` to reuse the saved results.

### Automatically choose the number of features

```bash
python run_classifier_pipeline.py --auto_choose_k
```

This selects the number of features with the highest cross-validated macro F1-score from `sfs_subset_size_results.csv`.

### Show plots interactively

```bash
python run_classifier_pipeline.py --show_plots
```

By default, plots are saved but not displayed.

---

## Running Training Only

```bash
python training_classifier.py
```

Useful options:

```bash
python training_classifier.py --n_features 5
python training_classifier.py --auto_choose_k
python training_classifier.py --skip_sfs_subset_test
```

The training script saves the final selected features and best SVM hyperparameters to:

```text
data/trained_classifier_config.json
```

---

## Running Testing Only

After training has created `trained_classifier_config.json`, run:

```bash
python testing_classifier.py
```

This loads the selected features and best parameters, trains the final model on the full training dataset, and evaluates it once on the unseen test dataset.

Custom paths can be passed as:

```bash
python testing_classifier.py --training_csv data/training_features.csv --testing_csv data/testing_features.csv --config data/trained_classifier_config.json
```

---

## Output Files

| File | Description |
|---|---|
| `data/trained_classifier_config.json` | Selected features, number of features, best `C`, best `gamma`, and best CV macro F1 |
| `data/single_feature_results.csv` | Accuracy and macro F1 for each feature individually |
| `data/sfs_subset_size_results.csv` | Accuracy and macro F1 for increasing numbers of selected features |
| `figures/macro_f1_vs_number_of_features.png` | Macro F1 versus number of selected features |
| `figures/accuracy_vs_number_of_features.png` | Accuracy versus number of selected features |
| `figures/training_cv_confusion_matrix.png` | Training cross-validation confusion matrix |
| `figures/test_confusion_matrix.png` | Final unseen test-set confusion matrix |

---

## Method Summary

The training procedure uses subject-wise `GroupKFold` cross-validation. This keeps all recordings from the same subject together in either the training fold or validation fold, preventing subject leakage.

The model is an RBF-kernel SVM inside a scikit-learn pipeline:

```text
StandardScaler + SVC(kernel="rbf")
```

The scaler is inside the pipeline so that feature standardization is learned only from the training fold during cross-validation.

Feature selection is performed using Sequential Forward Selection. The final selected feature subset is then used for SVM hyperparameter tuning with grid search over `C` and `gamma`.

The test dataset is only used at the end for the final independent evaluation.

---

## Recommended Workflow

For the first full run:

```bash
python run_classifier_pipeline.py --run_sfs_subset_test
```

For later runs:

```bash
python run_classifier_pipeline.py
```

To try a different number of selected features:

```bash
python run_classifier_pipeline.py --n_features 6
```

To let the script choose the feature count automatically:

```bash
python run_classifier_pipeline.py --auto_choose_k
```

---

## Notes

The unseen testing dataset should not be used for feature selection, hyperparameter tuning, or deciding which model to use. Training cross-validation results are used for model development. The final test results are the independent estimate of model performance.
