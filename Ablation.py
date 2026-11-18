
import pandas as pd
import numpy as np
import os
import joblib
import time

from pathlib import Path

from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
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

from sklearn.inspection import permutation_importance

# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "results" / "preprocessing"

OUTPUT_DIR = PROJECT_DIR / "results" / "ablation"

MODEL_DIR = PROJECT_DIR / "models" / "ablation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "label"

# ==========================================
# 2. LOAD PREPROCESSED DATA
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

print("Train Shape:", train_data.shape)
print("Validation Shape:", val_data.shape)
print("Test Shape:", test_data.shape)

# ==========================================
# 3. SEPARATE FEATURES AND TARGET
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

    if data[FEATURES].isna().any().any():
        raise ValueError(
            f"{name} contains missing feature values."
        )

X_train = train_data[FEATURES]
y_train = train_data[TARGET]

X_val = val_data[FEATURES]
y_val = val_data[TARGET]

X_test = test_data[FEATURES]
y_test = test_data[TARGET]

print("\nTotal Features:", len(FEATURES))

# ==========================================
# 4. DEFINE FEATURE GROUPS
# ==========================================

URL_FEATURES = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfDegitsInURL",
    "NoOfQMarkInURL",
    "NoOfOtherSpecialCharsInURL",
    "IsHTTPS"
]

WEBPAGE_FEATURES = [
    "DomainTitleMatchScore",
    "HasFavicon",
    "IsResponsive",
    "HasExternalFormSubmit",
    "HasPasswordField",
    "NoOfSelfRef",
    "NoOfExternalRef"
]

ENGINEERED_FEATURES = [
    "digit_density",
    "special_char_density",
    "obfuscation_density",
    "subdomain_density",
    "external_ref_ratio",
    "total_reference_count",
    "risky_form_combo",
    "log_url_length",
    "log_external_refs",
    "log_self_refs"
]

# Check feature availability

for group_name, group in [
    ("URL", URL_FEATURES),
    ("WEBPAGE", WEBPAGE_FEATURES),
    ("ENGINEERED", ENGINEERED_FEATURES)
]:

    missing = [
        col for col in group
        if col not in FEATURES
    ]

    if missing:
        raise ValueError(
            f"Missing {group_name} features: {missing}"
        )

ALL_FEATURES = (
    URL_FEATURES +
    WEBPAGE_FEATURES +
    ENGINEERED_FEATURES
)

# ==========================================
# 5. DEFINE MODELS
# ==========================================

def create_et():

    return ExtraTreesClassifier(
        n_estimators=200,
        max_depth=25,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )


def create_hgb():

    return HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        max_leaf_nodes=31,
        random_state=42
    )


MODEL_FACTORIES = {
    "Extra Trees": create_et,
    "Hist Gradient Boosting": create_hgb
}

# ==========================================
# 6. EVALUATION FUNCTION
# ==========================================

def evaluate_model(
    model,
    X_data,
    y_data,
    model_name,
    config_name
):

    predictions = model.predict(X_data)

    probabilities = model.predict_proba(X_data)

    accuracy = accuracy_score(
        y_data,
        predictions
    )

    precision = precision_score(
        y_data,
        predictions,
        pos_label=0,
        zero_division=0
    )

    recall = recall_score(
        y_data,
        predictions,
        pos_label=0,
        zero_division=0
    )

    phishing_f1 = f1_score(
        y_data,
        predictions,
        pos_label=0,
        zero_division=0
    )

    macro_f1 = f1_score(
        y_data,
        predictions,
        average="macro",
        zero_division=0
    )

    auc = roc_auc_score(
        y_data,
        probabilities[:, 0],
        labels=[0, 1]
    )

    return {

        "Configuration": config_name,
        "Model": model_name,
        "Features": len(X_data.columns),
        "Accuracy": accuracy,
        "Phishing_Precision": precision,
        "Phishing_Recall": recall,
        "Phishing_F1": phishing_f1,
        "Macro_F1": macro_f1,
        "ROC_AUC": auc

    }


# ==========================================
# 7. DEFINE ABLATION CONFIGURATIONS
# ==========================================

CONFIGURATIONS = {

    "all_28": ALL_FEATURES,

    "original_18": (
        URL_FEATURES +
        WEBPAGE_FEATURES
    ),

    "url_only_11": URL_FEATURES,

    "webpage_only_7": WEBPAGE_FEATURES,

    "engineered_only_10": ENGINEERED_FEATURES,

    "url_plus_engineered": (
        URL_FEATURES +
        ENGINEERED_FEATURES
    ),

    "webpage_plus_engineered": (
        WEBPAGE_FEATURES +
        ENGINEERED_FEATURES
    )

}

# ==========================================
# 8. TRAIN AND VALIDATE ABLATION MODELS
# ==========================================

validation_results = []

print("\n========== ABLATION STUDY STARTED ==========\n")

for config_name, selected_features in CONFIGURATIONS.items():

    print("\nConfiguration:", config_name)
    print("Number of Features:", len(selected_features))

    for model_name, model_factory in MODEL_FACTORIES.items():

        model = model_factory()

        start_time = time.time()

        model.fit(
            X_train[selected_features],
            y_train
        )

        training_time = time.time() - start_time

        result = evaluate_model(
            model,
            X_val[selected_features],
            y_val,
            model_name,
            config_name
        )

        result["Training_Time_Seconds"] = training_time

        validation_results.append(result)

        print(
            model_name,
            "| Validation Macro F1:",
            round(result["Macro_F1"], 4),
            "| Accuracy:",
            round(result["Accuracy"], 4)
        )

# Save validation results

validation_df = pd.DataFrame(validation_results)

validation_df = validation_df.sort_values(
    by="Macro_F1",
    ascending=False
)

validation_df.to_csv(
    OUTPUT_DIR / "ablation_validation_results.csv",
    index=False
)

print("\n========== VALIDATION RESULTS ==========\n")

print(
    validation_df.round(4).to_string(index=False)
)

# ==========================================
# 9. CORRELATION ANALYSIS
# ==========================================

print("\n========== CORRELATION ANALYSIS ==========\n")

correlation = X_train.corr().abs()

high_corr_pairs = []

for i in range(len(FEATURES)):

    for j in range(i + 1, len(FEATURES)):

        feature_a = FEATURES[i]
        feature_b = FEATURES[j]

        value = correlation.loc[
            feature_a,
            feature_b
        ]

        if value > 0.95:

            high_corr_pairs.append({

                "Feature_1": feature_a,
                "Feature_2": feature_b,
                "Correlation": value

            })

corr_df = pd.DataFrame(high_corr_pairs)

if not corr_df.empty:

    corr_df = corr_df.sort_values(
        by="Correlation",
        ascending=False
    )

    print(corr_df.round(4).to_string(index=False))

else:

    print("No feature pairs above 0.95 correlation.")

corr_df.to_csv(
    OUTPUT_DIR / "high_correlation_features.csv",
    index=False
)

# ==========================================
# 10. PERMUTATION IMPORTANCE
# ==========================================

print("\n========== PERMUTATION IMPORTANCE ==========\n")

importance_model = create_et()

importance_model.fit(
    X_train[ALL_FEATURES],
    y_train
)

# Use validation data for feature importance

sample_size = min(
    12000,
    len(X_val)
)

rng = np.random.RandomState(42)

sample_positions = rng.choice(
    len(X_val),
    size=sample_size,
    replace=False
)

X_val_sample = X_val.iloc[
    sample_positions
][ALL_FEATURES]

y_val_sample = y_val.iloc[
    sample_positions
]

pi = permutation_importance(

    importance_model,

    X_val_sample,

    y_val_sample,

    scoring="f1_macro",

    n_repeats=3,

    random_state=42,

    n_jobs=-1

)

importance_df = pd.DataFrame({

    "Feature": ALL_FEATURES,

    "Importance_Mean": pi.importances_mean,

    "Importance_Std": pi.importances_std

})

importance_df = importance_df.sort_values(
    by="Importance_Mean",
    ascending=False
)

importance_df.to_csv(
    OUTPUT_DIR / "permutation_importance.csv",
    index=False
)

print(
    importance_df.round(4).to_string(index=False)
)

# ==========================================
# 11. TOP FEATURE EXPERIMENTS
# ==========================================

print("\n========== TOP FEATURE EXPERIMENTS ==========\n")

top_results = []

top_features = importance_df[
    "Feature"
].tolist()

for k in [3, 5, 8, 10, 12, 15]:

    selected_features = top_features[:k]

    for model_name, model_factory in MODEL_FACTORIES.items():

        model = model_factory()

        start_time = time.time()

        model.fit(
            X_train[selected_features],
            y_train
        )

        training_time = time.time() - start_time

        result = evaluate_model(
            model,
            X_val[selected_features],
            y_val,
            model_name,
            f"top_{k}_features"
        )

        result["Training_Time_Seconds"] = training_time

        top_results.append(result)

        print(
            f"Top {k} | {model_name} | "
            f"Macro F1: {result['Macro_F1']:.4f}"
        )

top_df = pd.DataFrame(top_results)

top_df.to_csv(
    OUTPUT_DIR / "top_features_validation_results.csv",
    index=False
)

# ==========================================
# 12. COMBINE ALL VALIDATION RESULTS
# ==========================================

all_validation_results = pd.concat(
    [
        validation_df,
        top_df
    ],
    ignore_index=True
)

all_validation_results = all_validation_results.sort_values(
    by="Macro_F1",
    ascending=False
)

all_validation_results.to_csv(
    OUTPUT_DIR / "complete_validation_comparison.csv",
    index=False
)

# Select best configuration using validation Macro F1

best_result = all_validation_results.iloc[0]

best_config = best_result["Configuration"]

best_model_name = best_result["Model"]

# Resolve selected feature list

if best_config in CONFIGURATIONS:

    best_features = CONFIGURATIONS[best_config]

else:

    k = int(best_config.split("_")[1])

    best_features = top_features[:k]

print("\n========== BEST VALIDATION CONFIGURATION ==========")

print("Best Configuration:", best_config)
print("Best Model:", best_model_name)
print("Selected Features:", len(best_features))
print("Validation Macro F1:", round(best_result["Macro_F1"], 4))

# ==========================================
# 13. FINAL MODEL TRAINING
# ==========================================

print("\n========== FINAL MODEL TRAINING ==========\n")

if best_model_name == "Extra Trees":

    final_model = create_et()

else:

    final_model = create_hgb()

# Train on training data only

final_model.fit(
    X_train[best_features],
    y_train
)

# Save selected model and features

joblib.dump(
    final_model,
    MODEL_DIR / "best_ablation_model.pkl"
)

joblib.dump(
    best_features,
    MODEL_DIR / "best_ablation_features.pkl"
)

# ==========================================
# 14. FINAL TEST EVALUATION
# ==========================================

print("\n========== FINAL TEST EVALUATION ==========\n")

test_result = evaluate_model(

    final_model,

    X_test[best_features],

    y_test,

    best_model_name,

    best_config

)

print("Best Model:", best_model_name)
print("Best Configuration:", best_config)

print("Test Accuracy:", round(test_result["Accuracy"], 4))
print("Phishing Precision:", round(test_result["Phishing_Precision"], 4))
print("Phishing Recall:", round(test_result["Phishing_Recall"], 4))
print("Phishing F1:", round(test_result["Phishing_F1"], 4))
print("Macro F1:", round(test_result["Macro_F1"], 4))
print("ROC-AUC:", round(test_result["ROC_AUC"], 4))

# ==========================================
# 15. CONFUSION MATRIX
# ==========================================

test_predictions = final_model.predict(
    X_test[best_features]
)

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
    OUTPUT_DIR / "final_confusion_matrix.csv"
)

print("\nConfusion Matrix:")
print(cm_df)

# ==========================================
# 16. CLASSIFICATION REPORT
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
    OUTPUT_DIR / "final_classification_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(report)

# ==========================================
# 17. SAVE FINAL RESULTS
# ==========================================

final_df = pd.DataFrame([test_result])

final_df["Best_Model"] = best_model_name

final_df["Best_Configuration"] = best_config

final_df["Selected_Features"] = len(best_features)

final_df.to_csv(
    OUTPUT_DIR / "final_test_results.csv",
    index=False
)

pd.DataFrame({
    "Selected_Feature": best_features
}).to_csv(
    OUTPUT_DIR / "best_selected_features.csv",
    index=False
)

# ==========================================
# 18. FINISHED
# ==========================================

print("\n==========================================")
print("ABLATION STUDY COMPLETED SUCCESSFULLY!")
print("Best Model:", best_model_name)
print("Best Configuration:", best_config)
print("Test Macro F1:", round(test_result["Macro_F1"], 4))
print("==========================================")
