
import pandas as pd
import numpy as np
import os
import tldextract
import joblib

from pathlib import Path
from urllib.parse import urlsplit
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# --------------------------------
# 1. Load Dataset
# --------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    PROJECT_DIR
    / "phishing_project"
    / "phiusiil+phishing+url+dataset"
    / "PhiUSIIL_Phishing_URL_Dataset.csv"
)

OUTPUT_DIR = PROJECT_DIR / "results" / "preprocessing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Please check the dataset file location."
    )

df = pd.read_csv(DATA_PATH)

print("Original Dataset Shape:", df.shape)

# --------------------------------
# 2. Select 18 Original Features
# --------------------------------

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

FEATURES = URL_FEATURES + WEBPAGE_FEATURES
TARGET = "label"

REQUIRED_COLUMNS = FEATURES + [
    TARGET, "URL", "Domain"
]

missing_columns = [
    col for col in REQUIRED_COLUMNS
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

# --------------------------------
# 3. Clean Dataset
# --------------------------------

df = df.dropna(
    subset=["URL", "Domain", TARGET]
).copy()

df[TARGET] = pd.to_numeric(
    df[TARGET], errors="coerce"
)

df = df.dropna(subset=[TARGET]).copy()

if not (df[TARGET] % 1 == 0).all():
    raise ValueError("Target contains non-integer labels.")

df[TARGET] = df[TARGET].astype(int)

if not set(df[TARGET].unique()).issubset({0, 1}):
    raise ValueError("Label must contain only 0 and 1.")

# Remove URLs having conflicting labels

conflicting_urls = (
    df.groupby("URL")[TARGET].nunique()
)

bad_urls = conflicting_urls[
    conflicting_urls > 1
].index

df = df[~df["URL"].isin(bad_urls)].copy()

# Remove duplicate URLs

df = df.drop_duplicates(
    subset=["URL"],
    keep="first"
).copy()

print("After Cleaning:", df.shape)

# --------------------------------
# 4. Numeric Conversion
# --------------------------------

for col in FEATURES:
    df[col] = pd.to_numeric(
        df[col], errors="coerce"
    )

df[FEATURES] = df[FEATURES].replace(
    [np.inf, -np.inf], np.nan
)

# Stop if any feature is completely empty

all_missing = [
    col for col in FEATURES
    if df[col].isna().all()
]

if all_missing:
    raise ValueError(
        f"These features contain only missing values: {all_missing}"
    )

# --------------------------------
# 5. Registered Domain Extraction
# --------------------------------

extractor = tldextract.TLDExtract(
    suffix_list_urls=None
)

def get_registered_domain(value):

    value = str(value).strip().lower()

    if not value:
        return None

    if "://" not in value:
        value = "//" + value

    host = urlsplit(value).hostname

    if not host:
        return None

    host = host.lower().strip(".")

    result = extractor(host)

    if result.domain and result.suffix:
        return result.domain + "." + result.suffix

    # Fallback for IP addresses or special hostnames
    return host

df["registered_domain"] = (
    df["Domain"].apply(get_registered_domain)
)

if df["registered_domain"].isna().any():
    raise ValueError(
        "Some domains could not be extracted. "
        "Check the Domain column."
    )

print(
    "Unique Registered Domains:",
    df["registered_domain"].nunique()
)

# --------------------------------
# 6. Feature Engineering
# --------------------------------

epsilon = 1e-6

df["digit_density"] = (
    df["NoOfDegitsInURL"] /
    (df["URLLength"] + epsilon)
)

df["special_char_density"] = (
    df["NoOfOtherSpecialCharsInURL"] /
    (df["URLLength"] + epsilon)
)

df["obfuscation_density"] = (
    df["NoOfObfuscatedChar"] /
    (df["URLLength"] + epsilon)
)

df["subdomain_density"] = (
    df["NoOfSubDomain"] /
    (df["DomainLength"] + epsilon)
)

df["total_reference_count"] = (
    df["NoOfSelfRef"] +
    df["NoOfExternalRef"]
)

df["external_ref_ratio"] = (
    df["NoOfExternalRef"] /
    (df["total_reference_count"] + epsilon)
)

df["risky_form_combo"] = (
    df["HasExternalFormSubmit"] *
    df["HasPasswordField"]
)

df["log_url_length"] = np.log1p(
    df["URLLength"].clip(lower=0)
)

df["log_external_refs"] = np.log1p(
    df["NoOfExternalRef"].clip(lower=0)
)

df["log_self_refs"] = np.log1p(
    df["NoOfSelfRef"].clip(lower=0)
)

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

ALL_FEATURES = FEATURES + ENGINEERED_FEATURES

print("\nOriginal Selected Features:", len(FEATURES))
print("Engineered Features:", len(ENGINEERED_FEATURES))
print("Total Input Features:", len(ALL_FEATURES))

# --------------------------------
# 7. Validate Binary Features
# --------------------------------

BINARY_FEATURES = [
    "IsDomainIP",
    "HasObfuscation",
    "IsHTTPS",
    "HasFavicon",
    "IsResponsive",
    "HasExternalFormSubmit",
    "HasPasswordField",
    "risky_form_combo"
]

for col in BINARY_FEATURES:
    values = set(df[col].dropna().unique())

    if not values.issubset({0, 1}):
        raise ValueError(
            f"{col} is not binary. Found values: {values}"
        )

CONTINUOUS_FEATURES = [
    col for col in ALL_FEATURES
    if col not in BINARY_FEATURES
]

# Save raw engineered dataset

df[ALL_FEATURES + [TARGET]].to_csv(
    OUTPUT_DIR / "engineered_dataset.csv",
    index=False
)

# --------------------------------
# 8. Group-based Train/Validation/Test Split
# --------------------------------

X = df[ALL_FEATURES].copy()
y = df[TARGET].copy()
groups = df["registered_domain"]

# First: 15% Test

split1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.15,
    random_state=42
)

train_val_idx, test_idx = next(
    split1.split(X, y, groups)
)

# Second: 15% Validation overall

X_train_val = X.iloc[train_val_idx]
y_train_val = y.iloc[train_val_idx]
groups_train_val = groups.iloc[train_val_idx]

validation_ratio = 0.15 / 0.85

split2 = GroupShuffleSplit(
    n_splits=1,
    test_size=validation_ratio,
    random_state=42
)

train_rel_idx, val_rel_idx = next(
    split2.split(
        X_train_val,
        y_train_val,
        groups_train_val
    )
)

train_idx = train_val_idx[train_rel_idx]
val_idx = train_val_idx[val_rel_idx]

X_train = X.iloc[train_idx].copy()
X_val = X.iloc[val_idx].copy()
X_test = X.iloc[test_idx].copy()

y_train = y.iloc[train_idx].copy()
y_val = y.iloc[val_idx].copy()
y_test = y.iloc[test_idx].copy()

# --------------------------------
# 9. Verify Domain Separation
# --------------------------------

train_domains = set(groups.iloc[train_idx])
val_domains = set(groups.iloc[val_idx])
test_domains = set(groups.iloc[test_idx])

assert train_domains.isdisjoint(val_domains)
assert train_domains.isdisjoint(test_domains)
assert val_domains.isdisjoint(test_domains)

print("\nDomain overlap check: PASSED")

# --------------------------------
# 10. Missing Value Imputation
# --------------------------------

imputer = SimpleImputer(strategy="median")

X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=ALL_FEATURES,
    index=X_train.index
)

X_val = pd.DataFrame(
    imputer.transform(X_val),
    columns=ALL_FEATURES,
    index=X_val.index
)

X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=ALL_FEATURES,
    index=X_test.index
)

# --------------------------------
# 11. Normalization
# --------------------------------

scaler = MinMaxScaler()

X_train[CONTINUOUS_FEATURES] = (
    scaler.fit_transform(
        X_train[CONTINUOUS_FEATURES]
    )
)

X_val[CONTINUOUS_FEATURES] = (
    scaler.transform(
        X_val[CONTINUOUS_FEATURES]
    )
)

X_test[CONTINUOUS_FEATURES] = (
    scaler.transform(
        X_test[CONTINUOUS_FEATURES]
    )
)

# --------------------------------
# 12. Save Processed Datasets
# --------------------------------

train_data = X_train.copy()
train_data[TARGET] = y_train

val_data = X_val.copy()
val_data[TARGET] = y_val

test_data = X_test.copy()
test_data[TARGET] = y_test

train_data.to_csv(
    OUTPUT_DIR / "train_data_scaled.csv",
    index=False
)

val_data.to_csv(
    OUTPUT_DIR / "validation_data_scaled.csv",
    index=False
)

test_data.to_csv(
    OUTPUT_DIR / "test_data_scaled.csv",
    index=False
)

joblib.dump(
    imputer,
    OUTPUT_DIR / "imputer.pkl"
)

joblib.dump(
    scaler,
    OUTPUT_DIR / "scaler.pkl"
)

# --------------------------------
# 13. Save Split Summary
# --------------------------------

summary = pd.DataFrame({
    "Dataset": ["Train", "Validation", "Test"],

    "Rows": [
        len(train_data),
        len(val_data),
        len(test_data)
    ],

    "Phishing (0)": [
        int((y_train == 0).sum()),
        int((y_val == 0).sum()),
        int((y_test == 0).sum())
    ],

    "Legitimate (1)": [
        int((y_train == 1).sum()),
        int((y_val == 1).sum()),
        int((y_test == 1).sum())
    ],

    "Unique Domains": [
        len(train_domains),
        len(val_domains),
        len(test_domains)
    ]
})

summary.to_csv(
    OUTPUT_DIR / "split_summary.csv",
    index=False
)

print("\nFinal Dataset Shapes:")
print("Train:", train_data.shape)
print("Validation:", val_data.shape)
print("Test:", test_data.shape)

print("\nSplit Summary:")
print(summary.to_string(index=False))

print("\nPreprocessing Completed Successfully!")
