
import pandas as pd
import numpy as np
import os
import joblib
import time

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# --------------------------------
# 1. Paths
# --------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "results" / "preprocessing"
MODEL_DIR = PROJECT_DIR / "models" / "trained"
RESULT_DIR = PROJECT_DIR / "results" / "model_training"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "label"

# --------------------------------
# 2. Load Processed Datasets
# --------------------------------

train_data = pd.read_csv(
    DATA_DIR / "train_data_scaled.csv"
)

val_data = pd.read_csv(
    DATA_DIR / "validation_data_scaled.csv"
)

test_data = pd.read_csv(
    DATA_DIR / "test_data_scaled.csv"
)

print("Datasets Loaded Successfully!")

print("Train Shape:", train_data.shape)
print("Validation Shape:", val_data.shape)
print("Test Shape:", test_data.shape)

# --------------------------------
# 3. Separate X and y
# --------------------------------

if TARGET not in train_data.columns:
    raise ValueError("Target column not found!")

feature_columns = [
    col for col in train_data.columns
    if col != TARGET
]

for name, data in [
    ("Validation", val_data),
    ("Test", test_data)
]:
    if set(data.columns) != set(train_data.columns):
        raise ValueError(
            f"{name} columns do not match training columns."
        )

    data = data[train_data.columns]

X_train = train_data[feature_columns]
y_train = train_data[TARGET]

X_val = val_data[feature_columns]
y_val = val_data[TARGET]

X_test = test_data[feature_columns]
y_test = test_data[TARGET]

print("\nNumber of Input Features:", len(feature_columns))

# --------------------------------
# 4. Define Models
# --------------------------------

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=25,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=25,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200,
        max_depth=25,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ),

    "AdaBoost": AdaBoostClassifier(
        n_estimators=100,
        learning_rate=0.5,
        random_state=42
    ),

    "Support Vector Machine": LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=5000,
        random_state=42
    )
}

# --------------------------------
# 5. Train and Validate Models
# --------------------------------

validation_results = []

trained_models = {}

print("\n========== MODEL TRAINING STARTED ==========\n")

for model_name, model in models.items():

    print(f"Training: {model_name}")

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    trained_models[model_name] = model

    # Validation prediction

    val_predictions = model.predict(X_val)

    # Get phishing scores for ROC-AUC
    # Phishing label = 0

    if hasattr(model, "predict_proba"):

        val_scores = model.predict_proba(X_val)[
            :, list(model.classes_).index(0)
        ]

    else:

        val_scores = -model.decision_function(X_val)

    # Metrics

    accuracy = accuracy_score(
        y_val,
        val_predictions
    )

    precision = precision_score(
        y_val,
        val_predictions,
        pos_label=0,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        val_predictions,
        pos_label=0,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        val_predictions,
        pos_label=0,
        zero_division=0
    )

    macro_f1 = f1_score(
        y_val,
        val_predictions,
        average="macro",
        zero_division=0
    )

    y_val_phishing = (y_val == 0).astype(int)

    auc = roc_auc_score(
        y_val_phishing,
        val_scores
    )

    validation_results.append({

        "Model": model_name,
        "Accuracy": accuracy,
        "Phishing_Precision": precision,
        "Phishing_Recall": recall,
        "Phishing_F1": f1,
        "Macro_F1": macro_f1,
        "ROC_AUC": auc,
        "Training_Time_Seconds": training_time

    })

    print("Accuracy:", round(accuracy, 4))
    print("Phishing Precision:", round(precision, 4))
    print("Phishing Recall:", round(recall, 4))
    print("Phishing F1:", round(f1, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("ROC-AUC:", round(auc, 4))
    print("Training Time:", round(training_time, 2), "seconds")

    # Save each trained model

    safe_name = model_name.lower().replace(" ", "_")

    joblib.dump(
        model,
        MODEL_DIR / f"{safe_name}.pkl"
    )

    print("-" * 50)

# --------------------------------
# 6. Compare Validation Results
# --------------------------------

results_df = pd.DataFrame(validation_results)

results_df = results_df.sort_values(
    by="Macro_F1",
    ascending=False
)

results_df.to_csv(
    RESULT_DIR / "validation_model_comparison.csv",
    index=False
)

print("\n========== VALIDATION COMPARISON ==========\n")

print(
    results_df.to_string(index=False)
)

# --------------------------------
# 7. Select Best Model
# --------------------------------

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("\nBest Model:", best_model_name)

joblib.dump(
    best_model,
    MODEL_DIR / "best_model.pkl"
)

# --------------------------------
# 8. Final Test Evaluation
# --------------------------------

print("\n========== FINAL TEST EVALUATION ==========\n")

test_predictions = best_model.predict(X_test)

# Get phishing scores for ROC-AUC

if hasattr(best_model, "predict_proba"):

    test_scores = best_model.predict_proba(X_test)[
        :, list(best_model.classes_).index(0)
    ]

else:

    test_scores = -best_model.decision_function(X_test)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_precision = precision_score(
    y_test,
    test_predictions,
    pos_label=0,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_predictions,
    pos_label=0,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    pos_label=0,
    zero_division=0
)

test_macro_f1 = f1_score(
    y_test,
    test_predictions,
    average="macro",
    zero_division=0
)

y_test_phishing = (y_test == 0).astype(int)

test_auc = roc_auc_score(
    y_test_phishing,
    test_scores
)

print("Selected Model:", best_model_name)
print("Test Accuracy:", round(test_accuracy, 4))
print("Phishing Precision:", round(test_precision, 4))
print("Phishing Recall:", round(test_recall, 4))
print("Phishing F1:", round(test_f1, 4))
print("Macro F1:", round(test_macro_f1, 4))
print("ROC-AUC:", round(test_auc, 4))

# --------------------------------
# 9. Confusion Matrix
# --------------------------------

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
)

cm_df = pd.DataFrame(
    cm,
    index=["Actual Phishing", "Actual Legitimate"],
    columns=["Predicted Phishing", "Predicted Legitimate"]
)

cm_df.to_csv(
    RESULT_DIR / "best_model_confusion_matrix.csv"
)

print("\nConfusion Matrix:")
print(cm_df)

# --------------------------------
# 10. Classification Report
# --------------------------------

report = classification_report(
    y_test,
    test_predictions,
    labels=[0, 1],
    target_names=["Phishing", "Legitimate"],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

report_df.to_csv(
    RESULT_DIR / "best_model_classification_report.csv"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["Phishing", "Legitimate"],
        zero_division=0
    )
)

# --------------------------------
# 11. Save Final Test Results
# --------------------------------

final_results = pd.DataFrame([{

    "Best_Model": best_model_name,
    "Test_Accuracy": test_accuracy,
    "Phishing_Precision": test_precision,
    "Phishing_Recall": test_recall,
    "Phishing_F1": test_f1,
    "Macro_F1": test_macro_f1,
    "ROC_AUC": test_auc

}])

final_results.to_csv(
    RESULT_DIR / "final_test_results.csv",
    index=False
)

# --------------------------------
# 12. Final Message
# --------------------------------

print("\n======================================")
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("Total Models:", len(models))
print("Best Model:", best_model_name)
print("======================================")
