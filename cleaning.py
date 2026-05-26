import pandas as pd
import seaborn as sns
from scipy.stats import zscore


# Load Titanic dataset
titanic = sns.load_dataset("titanic")


print("\n===== ORIGINAL DATA =====")
print(titanic.head())


# ---------------------------------------------------
# 1. HANDLE MISSING VALUES
# ---------------------------------------------------

print("\n===== MISSING VALUES BEFORE CLEANING =====")
print(titanic.isnull().sum())


# Strategy 1: Fill missing age with mean
titanic["age"] = titanic["age"].fillna(titanic["age"].mean())


# Strategy 2: Fill missing embarked with mode
titanic["embarked"] = titanic["embarked"].fillna(
    titanic["embarked"].mode()[0]
)


# Strategy 3: Drop rows where deck is missing
titanic = titanic.dropna(subset=["deck"])


print("\n===== MISSING VALUES AFTER CLEANING =====")
print(titanic.isnull().sum())


# ---------------------------------------------------
# 2. DETECT OUTLIERS USING IQR
# ---------------------------------------------------

Q1 = titanic["fare"].quantile(0.25)
Q3 = titanic["fare"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


iqr_outliers = titanic[
    (titanic["fare"] < lower_bound) |
    (titanic["fare"] > upper_bound)
]

print("\n===== IQR OUTLIERS =====")
print(iqr_outliers[["fare"]])


# Remove IQR outliers
titanic = titanic[
    (titanic["fare"] >= lower_bound) &
    (titanic["fare"] <= upper_bound)
]


# ---------------------------------------------------
# 3. DETECT OUTLIERS USING Z-SCORE
# ---------------------------------------------------

titanic["zscore"] = zscore(titanic["fare"])

zscore_outliers = titanic[titanic["zscore"].abs() > 3]

print("\n===== Z-SCORE OUTLIERS =====")
print(zscore_outliers[["fare", "zscore"]])


# Remove z-score outliers
titanic = titanic[titanic["zscore"].abs() <= 3]


# ---------------------------------------------------
# 4. DATA TYPE CONVERSION
# ---------------------------------------------------

print("\n===== DATA TYPES BEFORE =====")
print(titanic.dtypes)


# Convert survived to string
titanic["survived"] = titanic["survived"].astype(str)


print("\n===== DATA TYPES AFTER =====")
print(titanic.dtypes)


# ---------------------------------------------------
# FINAL CLEAN DATA
# ---------------------------------------------------

print("\n===== CLEANED DATA =====")
print(titanic.head())