import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)

# ---------- 1. Load ----------
df = pd.read_csv("phishing_project/phiusiil+phishing+url+dataset/PhiUSIIL_Phishing_URL_Dataset.csv")
print("Shape:", df.shape)
print("Null:", df.isnull().sum().sum())
print(df["label"].value_counts())

# ---------- 2. Preprocessing ----------
df = df.drop_duplicates(subset="URL")
df = df.drop(columns=["FILENAME", "URL", "Domain", "Title", "TLD"])

X = df.drop(columns="label")
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
print("Train:", X_train.shape, "Test:", X_test.shape)

# ---------- 3. Models ----------
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                            n_jobs=-1, random_state=42),
}

# ---------- 4. Train + Evaluate ----------
results = []
for name, model in models.items():
    print("\n=====", name, "=====")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(classification_report(y_test, pred, target_names=["Phishing(0)", "Legitimate(1)"]))
    print(confusion_matrix(y_test, pred))
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision(phishing)": precision_score(y_test, pred, pos_label=0),
        "Recall(phishing)": recall_score(y_test, pred, pos_label=0),
        "F1(phishing)": f1_score(y_test, pred, pos_label=0),
    })

# ---------- 5. Comparison table ----------
res_df = pd.DataFrame(results).round(4)
print("\n", res_df.to_string(index=False))
res_df.to_csv("model_comparison.csv", index=False)

# ---------- 6. Cross-validation (Random Forest) ----------
cv = cross_val_score(models["Random Forest"], X_train, y_train, cv=5, scoring="f1", n_jobs=-1)
print("\nCV F1 scores:", cv.round(4), "Mean:", cv.mean().round(4))

# ---------- 7. Feature importance ----------
imp = pd.Series(models["Random Forest"].feature_importances_, index=X.columns)
print("\nTop 10 features:\n", imp.sort_values(ascending=False).head(10))

# ---------- 8. Save model ----------
joblib.dump(models["Random Forest"], "rf_model.pkl")

# ---------- 9. Sample prediction ----------
sample = X_test.iloc[[0]]
print("\nPrediction:", models["Random Forest"].predict(sample)[0],
      "| Actual:", y_test.iloc[0], "(1=Legitimate, 0=Phishing)")