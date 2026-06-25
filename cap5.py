import pandas as pd

print("CAPSTONE PROJECT 5 : BUSINESS-FIRST DATA SCIENCE PROJECT")

df = pd.read_excel("Telco_customer_churn.xlsx")

print("\nDataset Shape :", df.shape)

print("\nBusiness Problem")
print("The telecom company wants to reduce customer churn and improve customer retention.")

print("\nBusiness Objective")
print("1. Identify customers likely to churn.")
print("2. Understand factors affecting churn.")
print("3. Help the business improve retention.")
print("4. Increase Customer Lifetime Value.")

print("\nDataset Overview")
print("Rows :", len(df))
print("Columns :", df.shape[1])

print("\nMissing Values")
print(df.isnull().sum().sum())

print("\nDuplicate Records")
print(df.duplicated().sum())

print("\nAverage Monthly Charges :", round(df["Monthly Charges"].mean(),2))
print("Average Tenure :", round(df["Tenure Months"].mean(),2))
print("Average CLTV :", round(df["CLTV"].mean(),2))

print("\nChurn Distribution")
print(df["Churn Label"].value_counts())

print("\nImportant Features")

features = [
    "Tenure Months",
    "Monthly Charges",
    "Contract",
    "Internet Service",
    "Payment Method",
    "CLTV",
    "Churn Score"
]

for feature in features:
    print("-", feature)

print("\nData Science Solution")

print("Problem Type : Classification")
print("Target Variable : Churn Label")

print("\nSuggested Models")

models = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest"
]

for model in models:
    print("-", model)

print("\nExpected Output")

outputs = [
    "Customer Churn Prediction",
    "Customer Risk Score",
    "Retention Recommendation"
]

for output in outputs:
    print("-", output)

