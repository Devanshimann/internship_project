import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

print("=" * 70)
print("PROJECT 2 : PYTHON BASED DATA PROCESSING")
print("=" * 70)

df = pd.read_excel("Telco_customer_churn.xlsx")

print("\nDataset Shape :", df.shape)

print("\nFirst Five Rows")
print(df.head())

print("\nColumns")
print(df.columns.tolist())

df["Customer Value"] = df["Monthly Charges"] * df["Tenure Months"]

df["Tenure Category"] = pd.cut(
    df["Tenure Months"],
    bins=[0,12,24,48,72],
    labels=["New","Growing","Loyal","Very Loyal"],
    include_lowest=True
)

df["Spending Category"] = pd.cut(
    df["Monthly Charges"],
    bins=[0,40,70,100,150],
    labels=["Low","Medium","High","Premium"]
)

df["Premium Customer"] = np.where(
    df["Monthly Charges"] >= df["Monthly Charges"].median(),
    "Yes",
    "No"
)

df["Long Term Customer"] = np.where(
    df["Tenure Months"] >= 24,
    "Yes",
    "No"
)

df["Internet Customer"] = np.where(
    df["Internet Service"]=="No",
    "No",
    "Yes"
)

df["Risk Score"] = (
    (1-df["Tenure Months"]/df["Tenure Months"].max())*0.40+
    (df["Monthly Charges"]/df["Monthly Charges"].max())*0.30+
    (df["Churn Score"]/df["Churn Score"].max())*0.30
).round(2)

df["Risk Category"] = pd.cut(
    df["Risk Score"],
    bins=[0,0.35,0.60,1],
    labels=["Low","Medium","High"]
)

print("\nAverage Monthly Charges :",round(df["Monthly Charges"].mean(),2))
print("Average Tenure :",round(df["Tenure Months"].mean(),2))
print("Average CLTV :",round(df["CLTV"].mean(),2))

print("\nContract Distribution")
print(df["Contract"].value_counts())

print("\nInternet Service Distribution")
print(df["Internet Service"].value_counts())

print("\nRisk Category")
print(df["Risk Category"].value_counts())

print("\nPremium Customer")
print(df["Premium Customer"].value_counts())

print("\nProcessed Data")

print(
    df[
        [
            "CustomerID",
            "Monthly Charges",
            "Customer Value",
            "Risk Score",
            "Risk Category",
            "Premium Customer"
        ]
    ].head(10)
)

print("\n")
print("=" * 70)
print("PROJECT 3 : DATA CLEANING & PREPARATION")
print("=" * 70)

print("\nMissing Values Before Cleaning")

print(df.isnull().sum())

num_cols = df.select_dtypes(include=np.number).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

cat_cols = df.select_dtypes(include="object").columns

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

duplicates = df.duplicated().sum()

print("\nDuplicate Records :",duplicates)

df.drop_duplicates(inplace=True)

numeric_columns = [
    "Monthly Charges",
    "Total Charges",
    "CLTV",
    "Churn Score"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col],errors="coerce")

text_columns = [
    "Gender",
    "Internet Service",
    "Contract",
    "Payment Method",
    "Churn Label"
]

for col in text_columns:
    df[col] = df[col].str.strip()

df = df[df["Monthly Charges"]>=0]
df = df[df["Total Charges"]>=0]
df = df[df["Tenure Months"]>=0]
df = df[df["CLTV"]>=0]

print("\nOutlier Report")

outlier_columns = [
    "Monthly Charges",
    "Total Charges",
    "Tenure Months",
    "CLTV"
]

for col in outlier_columns:

    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)

    iqr = q3-q1

    lower = q1-1.5*iqr
    upper = q3+1.5*iqr

    outliers = len(df[(df[col]<lower)|(df[col]>upper)])

    print(col,":",outliers)

    df[col] = df[col].clip(lower,upper)

print("\nMissing Values After Cleaning")

print(df.isnull().sum())

print("\nFinal Shape :",df.shape)

print("\nData Types")

print(df.dtypes)

print("\nSummary Statistics")

print(df.describe())

print("\nUnique Values")

print("Gender :",df["Gender"].unique())
print("Contract :",df["Contract"].unique())
print("Internet Service :",df["Internet Service"].unique())
print("Payment Method :",df["Payment Method"].unique())
print("Churn Label :",df["Churn Label"].unique())

print("\nFirst Five Rows")

print(df.head())

print("\nProjects 2 and 3 Completed Successfully")
print("\n")
print("=" * 70)
print("PROJECT 4 : INSIGHT DISCOVERY THROUGH EDA")
print("=" * 70)

print("\nDataset Shape :", df.shape)

print("\nSummary Statistics")

print(df.describe())

print("\nChurn Distribution")

print(df["Churn Label"].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(data=df,x="Churn Label")
plt.title("Customer Churn Distribution")
plt.show()

plt.figure(figsize=(7,4))
sns.countplot(data=df,x="Contract")
plt.title("Contract Distribution")
plt.xticks(rotation=15)
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(data=df,x="Internet Service")
plt.title("Internet Service Distribution")
plt.show()

plt.figure(figsize=(7,4))
sns.histplot(df["Monthly Charges"],bins=30,kde=True)
plt.title("Monthly Charges Distribution")
plt.show()

plt.figure(figsize=(7,4))
sns.histplot(df["Tenure Months"],bins=30,kde=True)
plt.title("Tenure Distribution")
plt.show()

plt.figure(figsize=(7,5))
sns.boxplot(
    data=df,
    x="Churn Label",
    y="Monthly Charges"
)
plt.title("Monthly Charges vs Churn")
plt.show()

plt.figure(figsize=(7,5))
sns.boxplot(
    data=df,
    x="Churn Label",
    y="Tenure Months"
)
plt.title("Tenure vs Churn")
plt.show()

plt.figure(figsize=(8,5))
sns.countplot(
    data=df,
    x="Contract",
    hue="Churn Label"
)
plt.title("Contract vs Churn")
plt.xticks(rotation=15)
plt.show()

numeric = df.select_dtypes(include=["int64","float64"])

plt.figure(figsize=(10,7))
sns.heatmap(
    numeric.corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show()

print("\nBusiness Insights")

print("1. Month-to-month customers show higher churn.")
print("2. Customers with low tenure leave more often.")
print("3. Higher monthly charges are associated with churn.")
print("4. Long-term customers have better retention.")
print("5. Contract type plays an important role in customer retention.")

print("\n")
print("=" * 70)
print("PROJECT 5 : STATISTICAL DECISION SUPPORT")
print("=" * 70)

print("\nHypothesis 1")

print("H0 : Monthly Charges are not related to Churn")
print("H1 : Monthly Charges are related to Churn")

churn_yes = df[df["Churn Label"]=="Yes"]["Monthly Charges"]

churn_no = df[df["Churn Label"]=="No"]["Monthly Charges"]

t_stat,p_value = stats.ttest_ind(
    churn_yes,
    churn_no
)

print("\nT Statistic :",round(t_stat,3))
print("P Value :",round(p_value,5))

if p_value < 0.05:
    print("Reject Null Hypothesis")
    print("Monthly Charges significantly affect churn.")
else:
    print("Fail to Reject Null Hypothesis")

print("\n")

print("Hypothesis 2")

print("H0 : Contract Type and Churn are independent")

print("H1 : Contract Type and Churn are associated")

table = pd.crosstab(
    df["Contract"],
    df["Churn Label"]
)

chi2,p,dof,expected = stats.chi2_contingency(table)

print("\nChi Square :",round(chi2,3))

print("P Value :",round(p,5))

if p < 0.05:
    print("Reject Null Hypothesis")
    print("Contract type affects customer churn.")
else:
    print("Fail to Reject Null Hypothesis")

print("\n")

print("Hypothesis 3")

print("H0 : No relationship exists between Tenure and Monthly Charges")

print("H1 : Relationship exists")

corr,p = stats.pearsonr(
    df["Tenure Months"],
    df["Monthly Charges"]
)

print("\nCorrelation :",round(corr,3))

print("P Value :",round(p,5))

if p < 0.05:
    print("Reject Null Hypothesis")
    print("Significant correlation found.")
else:
    print("Fail to Reject Null Hypothesis")

print("\n")

print("Hypothesis 4")

print("H0 : CLTV is same for churned and retained customers")

print("H1 : CLTV differs between churned and retained customers")

cltv_yes = df[df["Churn Label"]=="Yes"]["CLTV"]

cltv_no = df[df["Churn Label"]=="No"]["CLTV"]

t_stat,p_value = stats.ttest_ind(
    cltv_yes,
    cltv_no
)

print("\nT Statistic :",round(t_stat,3))

print("P Value :",round(p_value,5))

if p_value < 0.05:
    print("Reject Null Hypothesis")
    print("CLTV differs significantly.")
else:
    print("Fail to Reject Null Hypothesis")

print("\nSummary")

print("T-Test : Monthly Charges vs Churn")

print("Chi-Square : Contract vs Churn")

print("Correlation : Tenure vs Monthly Charges")

print("T-Test : CLTV vs Churn")

print("\nProjects 4 and 5 Completed Successfully")
print("\n")
print("=" * 70)
print("PROJECT 6 : ML FEASIBILITY ASSESSMENT")
print("=" * 70)

print("\nBusiness Problem")
print("Predict whether a customer is likely to churn.")

print("\nTarget Variable")
print("Churn Label")

print("\nDataset Overview")

print("Rows :", len(df))
print("Columns :", df.shape[1])

print("\nMissing Values")

missing = df.isnull().sum().sum()

print(missing)

if missing == 0:
    print("Dataset has no missing values.")
else:
    print("Dataset requires preprocessing.")

print("\nDuplicate Records")

duplicates = df.duplicated().sum()

print(duplicates)

if duplicates == 0:
    print("No duplicate records found.")
else:
    print("Duplicate records should be removed.")

print("\nFeature Types")

numerical = df.select_dtypes(include=["int64","float64"]).columns

categorical = df.select_dtypes(include="object").columns

print("Numerical Features :", len(numerical))
print("Categorical Features :", len(categorical))

print("\nNumerical Columns")

for col in numerical:
    print("-", col)

print("\nCategorical Columns")

for col in categorical:
    print("-", col)

print("\nTarget Distribution")

print(df["Churn Label"].value_counts())

print("\nTarget Percentage")

print(
    round(
        df["Churn Label"].value_counts(normalize=True) * 100,
        2
    )
)

largest = df["Churn Label"].value_counts(normalize=True).max()

if largest < 0.80:
    print("\nClasses are reasonably balanced.")
else:
    print("\nDataset has class imbalance.")

print("\nDataset Size Assessment")

if len(df) >= 1000:
    print("Dataset size is sufficient for Machine Learning.")
else:
    print("More records are recommended.")

print("\nSuitable Machine Learning Technique")
print("\nRecommended Algorithms")

algorithms = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting",
    "XGBoost"
]

for algo in algorithms:
    print("-", algo)

print("\nEvaluation Metrics")

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]

for metric in metrics:
    print("-", metric)

print("\n")

print("PROJECTS 2 TO 6 COMPLETED SUCCESSFULLY")
