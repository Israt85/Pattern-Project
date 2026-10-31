import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ---------- Load + same preprocessing ----------
df = pd.read_csv("phishing_project/phiusiil+phishing+url+dataset/PhiUSIIL_Phishing_URL_Dataset.csv")
df = df.drop_duplicates(subset="URL")
df = df.drop(columns=["FILENAME", "URL", "Domain", "Title", "TLD"])

X = df.drop(columns="label")
y = df["label"]

# ---------- Ablation experiments ----------
strong = ["URLSimilarityIndex", "TLDLegitimateProb", "URLCharProb", "CharContinuationRate"]

experiments = {
    "A: All features": [],
    "B: Drop 4 strong features": strong,
    "C: Drop 4 strong + URLSimilarityIndex-like content (LineOfCode, NoOfExternalRef, NoOfSelfRef)":
        strong + ["LineOfCode", "NoOfExternalRef", "NoOfSelfRef"],
}

def get_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                                n_jobs=-1, random_state=42),
    }

results = []
for exp_name, drop_cols in experiments.items():
    Xe = X.drop(columns=drop_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        Xe, y, test_size=0.2, stratify=y, random_state=42)
    print("\n#####", exp_name, "| features:", Xe.shape[1])

    for name, model in get_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        print(name, "\n", confusion_matrix(y_test, pred))
        results.append({
            "Experiment": exp_name,
            "Model": name,
            "Features": Xe.shape[1],
            "Accuracy": accuracy_score(y_test, pred),
            "Precision(phishing)": precision_score(y_test, pred, pos_label=0),
            "Recall(phishing)": recall_score(y_test, pred, pos_label=0),
            "F1(phishing)": f1_score(y_test, pred, pos_label=0),
        })
        if name == "Random Forest":
            imp = pd.Series(model.feature_importances_, index=Xe.columns)
            print("Top 5 features:\n", imp.sort_values(ascending=False).head(5))

res_df = pd.DataFrame(results).round(4)
print("\n", res_df.to_string(index=False))
res_df.to_csv("ablation_results.csv", index=False)