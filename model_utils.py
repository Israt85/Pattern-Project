
import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

os.makedirs("models_25", exist_ok=True)
os.makedirs("results_25", exist_ok=True)


def run_model(model, model_name):

    print("\n================================")
    print("MODEL:", model_name)
    print("================================")

    # Load 25-feature datasets
    train_data = pd.read_csv("train_data.csv")
    test_data = pd.read_csv("test_data.csv")

    # Separate features and labels
    X_train = train_data.drop("label", axis=1)
    y_train = train_data["label"]

    X_test = test_data.drop("label", axis=1)
    y_test = test_data["label"]

    # Verify feature count
    print("Training Features:", X_train.shape[1])
    print("Testing Features:", X_test.shape[1])

    if X_train.shape[1] != 25:
        raise ValueError("Training dataset does not contain 25 features!")

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Train and Test features do not match!")

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Accuracy
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    # Precision, Recall, F1
    precision = precision_score(
        y_test, test_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test, test_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test, test_pred,
        average="macro",
        zero_division=0
    )

    # ROC-AUC
    roc_auc = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X_test)

        if len(model.classes_) == 2:

            positive_class = model.classes_[1]

            y_binary = (
                y_test == positive_class
            ).astype(int)

            roc_auc = roc_auc_score(
                y_binary,
                probabilities[:, 1]
            )

    # Train-Test Gap
    gap = train_acc - test_acc

    # Confusion Matrix
    cm = confusion_matrix(y_test, test_pred)

    # Classification Report
    report = classification_report(
        y_test,
        test_pred,
        zero_division=0
    )

    # Print results
    print("\nTrain Accuracy:", train_acc)
    print("Test Accuracy:", test_acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("ROC-AUC:", roc_auc)
    print("Train-Test Gap:", gap)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(report)

    # Save model
    joblib.dump(
        model,
        f"models_25/{model_name}.pkl"
    )

    # Save confusion matrix
    pd.DataFrame(cm).to_csv(
        f"results_25/{model_name}_confusion_matrix.csv",
        index=False
    )

    # Save report
    with open(
        f"results_25/{model_name}_report.txt",
        "w"
    ) as file:
        file.write(report)

    # Save comparison
    result = pd.DataFrame([{
        "Model": model_name,
        "Train Accuracy": train_acc,
        "Test Accuracy": test_acc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
        "Train-Test Gap": gap
    }])

    result_file = "results_25/model_comparison.csv"

    if os.path.exists(result_file):

        old = pd.read_csv(result_file)

        old = old[old["Model"] != model_name]

        result = pd.concat(
            [old, result],
            ignore_index=True
        )

    result.to_csv(result_file, index=False)

    print("\nModel saved successfully!")
