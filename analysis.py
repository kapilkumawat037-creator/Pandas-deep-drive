import pandas as pd
import seaborn as sns


# Load Titanic dataset
titanic = sns.load_dataset("titanic")


# Load SMS Spam dataset
sms_data = pd.read_csv(
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
    sep="\t",
    header=None,
    names=["label", "message"]
)


def inspect_data(dataframe, name):
    """
    Inspect dataset information.
    """

    print(f"\n===== {name} DATASET =====")

    print("\n--- INFO ---")
    print(dataframe.info())

    print("\n--- DESCRIBE ---")
    print(dataframe.describe(include="all"))

    print("\n--- MISSING VALUES ---")
    print(dataframe.isnull().sum())

    print("\n--- DATA TYPES ---")
    print(dataframe.dtypes)


# Titanic Inspection
inspect_data(titanic, "Titanic")


# SMS Dataset Inspection
inspect_data(sms_data, "SMS Spam")


print("\n--- Titanic Sex Counts ---")
print(titanic["sex"].value_counts())

print("\n--- SMS Label Counts ---")
print(sms_data["label"].value_counts())