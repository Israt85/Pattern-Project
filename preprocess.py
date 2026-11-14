
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import GroupShuffleSplit

# =================================
# STEP 1: LOAD DATASET
# =================================

df = pd.read_csv(
    "phishing_project/phiusiil+phishing+url+dataset/PhiUSIIL_Phishing_URL_Dataset.csv"
)

print("Original Shape:", df.shape)


# =================================
# STEP 2: REMOVE DUPLICATES
# =================================

df = df.drop_duplicates()

df = df.drop_duplicates(subset=["URL"])

print("After Duplicate Removal:", df.shape)


# =================================
# STEP 3: HANDLE NULL LABELS
# =================================

df = df.dropna(subset=["label"])

# Keep Domain for group splitting
groups = df["Domain"].copy()

# Target variable
y = df["label"].copy()


# =================================
# STEP 4: REMOVE UNNECESSARY COLUMNS
# =================================

remove_columns = [
    "FILENAME",
    "URL",
    "Domain",
    "Title",
    "TLD",
    "NoOfLettersInURL",
    "URLTitleMatchScore"
]

X = df.drop(columns=remove_columns + ["label"])

# Keep numerical features only
X = X.select_dtypes(include=np.number)

print("\nRemaining Features:", X.shape[1])

print("\nRemaining Columns:")
print(X.columns.tolist())


# =================================
# STEP 5: TRAIN TEST SPLIT
# =================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()

y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()

groups_train = groups.iloc[train_idx]
groups_test = groups.iloc[test_idx]

print("\nTrain-Test Split Completed!")


# =================================
# STEP 6: HANDLE MISSING VALUES
# =================================

# Calculate median from training data only
medians = X_train.median()

# Fill missing values
X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

print("\nMissing Values Handled!")


# =================================
# STEP 7: FEATURE SELECTION
# =================================

# Select 25 best features
selector = SelectKBest(
    score_func=mutual_info_classif,
    k=25
)

# Fit selector only on training data
X_train_selected = selector.fit_transform(
    X_train,
    y_train
)

# Apply same selection to testing data
X_test_selected = selector.transform(X_test)

# Get selected feature names
selected_features = X.columns[
    selector.get_support()
]

print("\nSelected 25 Features:")
print(selected_features.tolist())


# =================================
# STEP 8: CONVERT TO DATAFRAME
# =================================

X_train = pd.DataFrame(
    X_train_selected,
    columns=selected_features,
    index=X_train.index
)

X_test = pd.DataFrame(
    X_test_selected,
    columns=selected_features,
    index=X_test.index
)


# =================================
# STEP 9: SAVE FINAL DATASETS
# =================================

train_data = X_train.copy()
test_data = X_test.copy()

train_data["label"] = y_train
test_data["label"] = y_test

# Save training dataset
train_data.to_csv(
    "train_data.csv",
    index=False
)

# Save testing dataset
test_data.to_csv(
    "test_data.csv",
    index=False
)

# Save selected feature names
pd.Series(selected_features).to_csv(
    "selected_features.csv",
    index=False,
    header=["Feature"]
)


# =================================
# STEP 10: FINAL REPORT
# =================================

print("\n========== FINAL REPORT ==========")

print("Original Dataset Shape:", df.shape)

print("Original Numerical Features:", X.shape[1])

print("Selected Features:", len(selected_features))

print("Training Shape:", train_data.shape)

print("Testing Shape:", test_data.shape)

print(
    "Missing Values in Train:",
    train_data.isnull().sum().sum()
)

print(
    "Missing Values in Test:",
    test_data.isnull().sum().sum()
)

# Check domain overlap
domain_overlap = len(
    set(groups_train) & set(groups_test)
)

print("\nDomain Overlap:", domain_overlap)

print("\nTraining Labels:")
print(y_train.value_counts())

print("\nTesting Labels:")
print(y_test.value_counts())

print("\nSelected Features:")
for feature in selected_features:
    print("-", feature)

print("\nPreprocessing Completed Successfully!")
