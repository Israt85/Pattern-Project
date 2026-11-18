
import pandas as pd
import numpy as np
import joblib
import time

from pathlib import Path
from itertools import combinations

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "results" / "preprocessing"
MODEL_DIR = PROJECT_DIR / "models" / "hybrid"
RESULT_DIR = PROJECT_DIR / "results" / "hybrid"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "label"

# ==========================================
# 2. LOAD DATA
# ==========================================

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

print("Train:", train_data.shape)
print("Validation:", val_data.shape)
print("Test:", test_data.shape)

# ==========================================
# 3. SEPARATE FEATURES AND LABELS
# ==========================================

if TARGET not in train_data.columns:
    raise ValueError("Target column not found!")

FEATURES = [
    col for col in train_data.columns
    if col != TARGET
]

for name, data in [
    ("Validation", val_data),
    ("Test", test_data)
]:

    if set(data.columns) != set(train_data.columns):
        raise ValueError(
            f"{name} columns do not match training data."
        )

    if set(data[TARGET].unique()) != set(y for y in [0, 1]):
        raise ValueError(
            f"{name} must contain labels 0 and 1."
        )

X_train = train_data[FEATURES]
y_train = train_data[TARGET]

X_val = val_data[FEATURES]
y_val = val_data[TARGET]

X_test = test_data[FEATURES]
y_test = test_data[TARGET]

print("Total Features:", len(FEATURES))

# ==========================================
# 4. DEFINE SEVEN BASE MODELS
# ==========================================

def create_models():

    return {

        "ET": ExtraTreesClassifier(
            n_estimators=200,
            max_depth=25,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ),

        "RF": RandomForestClassifier(
            n_estimators=200,
            max_depth=25,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42
        ),

        "GB": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ),

        "LR": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),

        "DT": DecisionTreeClassifier(
            max_depth=25,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42
        ),

        "AB": AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.5,
            random_state=42
        ),

        "SVM": CalibratedClassifierCV(
            estimator=LinearSVC(
                C=1.0,
                class_weight="balanced",
                max_iter=5000,
                random_state=42
            ),
            method="sigmoid",
            cv=3,
            n_jobs=-1
        )
    }


# ==========================================
# 5. TRAIN ALL BASE MODELS ONCE
# ==========================================

base_models = create_models()

trained_models = {}

training_times = {}

print("\n========== BASE MODEL TRAINING ==========\n")

for model_name, model in base_models.items():

    print("Training:", model_name)

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    trained_models[model_name] = model

    training_times[model_name] = training_time

    print(
        "Training Time:",
        round(training_time, 2),
        "seconds"
    )

    # Save individual model

    joblib.dump(
        model,
        MODEL_DIR / f"{model_name.lower()}_model.pkl"
    )

    print("-" * 50)


# ==========================================
# 6. GET PHISHING PROBABILITIES
# ==========================================

def get_phishing_probabilities(model, X):

    probabilities = model.predict_proba(X)

    phishing_index = list(model.classes_).index(0)

    return probabilities[:, phishing_index]


# ==========================================
# 7. GET PROBABILITIES FOR ALL MODELS
# ==========================================

print("\n========== GENERATING VALIDATION SCORES ==========\n")

validation_probabilities = {}

for model_name, model in trained_models.items():

    validation_probabilities[model_name] = (
        get_phishing_probabilities(model, X_val)
    )

    print("Validation scores generated:", model_name)


# ==========================================
# 8. CREATE ALL THREE-MODEL COMBINATIONS
# ==========================================

MODEL_NAMES = list(trained_models.keys())

HYBRID_COMBINATIONS = list(
    combinations(MODEL_NAMES, 3)
)

print("\nTotal Base Models:", len(MODEL_NAMES))
print("Total Three-Model Hybrids:", len(HYBRID_COMBINATIONS))


# ==========================================
# 9. EVALUATION FUNCTION
# ==========================================

def calculate_metrics(y_true, predictions, phishing_scores, name):

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        pos_label=0,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        pos_label=0,
        zero_division=0
    )

    phishing_f1 = f1_score(
        y_true,
        predictions,
        pos_label=0,
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0
    )

    y_phishing = (y_true == 0).astype(int)

    auc = roc_auc_score(
        y_phishing,
        phishing_scores
    )

    return {

        "Model": name,
        "Accuracy": accuracy,
        "Phishing_Precision": precision,
        "Phishing_Recall": recall,
        "Phishing_F1": phishing_f1,
        "Macro_F1": macro_f1,
        "ROC_AUC": auc

    }


# ==========================================
# 10. EVALUATE ALL HYBRID COMBINATIONS
# ==========================================

validation_results = []

print("\n========== THREE-MODEL HYBRID TESTING ==========\n")

for model_combo in HYBRID_COMBINATIONS:

    hybrid_name = "_".join(model_combo)

    print("Testing:", hybrid_name)

    # Equal-weight probability averaging

    combined_phishing_scores = np.mean(
        [
            validation_probabilities[name]
            for name in model_combo
        ],
        axis=0
    )

    # Since there are two classes:
    # Label 0 = Phishing
    # Label 1 = Legitimate

    predictions = np.where(
        combined_phishing_scores >= 0.5,
        0,
        1
    )

    result = calculate_metrics(
        y_val,
        predictions,
        combined_phishing_scores,
        hybrid_name
    )

    result["Models"] = ", ".join(model_combo)

    result["Training_Time_Seconds"] = sum(
        training_times[name]
        for name in model_combo
    )

    validation_results.append(result)

    print("Accuracy:", round(result["Accuracy"], 5))
    print("Phishing Recall:", round(result["Phishing_Recall"], 5))
    print("Macro F1:", round(result["Macro_F1"], 5))
    print("ROC-AUC:", round(result["ROC_AUC"], 5))

    print("-" * 50)


# ==========================================
# 11. VALIDATION COMPARISON
# ==========================================

results_df = pd.DataFrame(validation_results)

results_df = results_df.sort_values(
    by="Macro_F1",
    ascending=False
)

results_df.to_csv(
    RESULT_DIR / "three_model_validation_comparison.csv",
    index=False
)

print("\n========== ALL HYBRID RESULTS ==========\n")

print(
    results_df.round(5).to_string(index=False)
)


# ==========================================
# 12. SELECT BEST HYBRID
# ==========================================

best_row = results_df.iloc[0]

best_hybrid_name = best_row["Model"]

best_models = best_hybrid_name.split("_")

print("\n====================================")
print("BEST THREE-MODEL HYBRID")
print("====================================")

print("Best Hybrid:", best_hybrid_name)

print("Selected Models:", best_models)

print("Validation Macro F1:", round(
    best_row["Macro_F1"], 5
))


# Save model bundle

best_bundle = {

    "models": trained_models,

    "selected_models": best_models,

    "features": FEATURES,

    "target": TARGET,

    "positive_label": 0,

    "method": "Equal Weight Soft Voting"

}

joblib.dump(
    best_bundle,
    MODEL_DIR / "best_three_model_hybrid.pkl"
)


# ==========================================
# 13. FINAL TEST EVALUATION
# ==========================================

print("\n========== FINAL TEST EVALUATION ==========\n")

# Generate test probabilities only for selected models

test_scores = []

for model_name in best_models:

    model = trained_models[model_name]

    scores = get_phishing_probabilities(
        model,
        X_test
    )

    test_scores.append(scores)


# Average the three model probabilities

final_phishing_scores = np.mean(
    test_scores,
    axis=0
)

test_predictions = np.where(
    final_phishing_scores >= 0.5,
    0,
    1
)


# Calculate final metrics

test_result = calculate_metrics(
    y_test,
    test_predictions,
    final_phishing_scores,
    best_hybrid_name
)


print("Best Hybrid:", best_hybrid_name)

print("Test Accuracy:", round(
    test_result["Accuracy"], 5
))

print("Phishing Precision:", round(
    test_result["Phishing_Precision"], 5
))

print("Phishing Recall:", round(
    test_result["Phishing_Recall"], 5
))

print("Phishing F1:", round(
    test_result["Phishing_F1"], 5
))

print("Macro F1:", round(
    test_result["Macro_F1"], 5
))

print("ROC-AUC:", round(
    test_result["ROC_AUC"], 5
))


# ==========================================
# 14. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
)

cm_df = pd.DataFrame(

    cm,

    index=[
        "Actual Phishing",
        "Actual Legitimate"
    ],

    columns=[
        "Predicted Phishing",
        "Predicted Legitimate"
    ]

)

cm_df.to_csv(
    RESULT_DIR / "three_model_confusion_matrix.csv"
)

print("\nConfusion Matrix:")
print(cm_df)


# ==========================================
# 15. CLASSIFICATION REPORT
# ==========================================

report = classification_report(

    y_test,

    test_predictions,

    labels=[0, 1],

    target_names=[
        "Phishing",
        "Legitimate"
    ],

    zero_division=0

)

print("\nClassification Report:")
print(report)

with open(
    RESULT_DIR / "three_model_classification_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(report)


# ==========================================
# 16. SAVE FINAL TEST RESULTS
# ==========================================

final_df = pd.DataFrame([test_result])

final_df["Selected_Models"] = ", ".join(best_models)

final_df.to_csv(
    RESULT_DIR / "three_model_final_test_results.csv",
    index=False
)


# ==========================================
# 17. SAVE TRAINING TIME
# ==========================================

training_df = pd.DataFrame(
    list(training_times.items()),
    columns=["Model", "Training_Time_Seconds"]
)

training_df.to_csv(
    RESULT_DIR / "base_model_training_times.csv",
    index=False
)


# ==========================================
# 18. COMPLETED
# ==========================================

print("\n==========================================")
print("THREE-MODEL HYBRID TRAINING COMPLETED!")
print("Total Base Models:", len(MODEL_NAMES))
print("Total Hybrid Combinations:", len(HYBRID_COMBINATIONS))
print("Best Hybrid:", best_hybrid_name)
print("Final Test Macro F1:", round(test_result["Macro_F1"], 5))
print("==========================================")
