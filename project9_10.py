import pandas as pd

print("=" * 70)
print("PROJECT 9 : AI USE-CASE DESIGN")
print("=" * 70)

df = pd.read_excel("Telco_customer_churn.xlsx")

print("\nDataset Shape :", df.shape)

print("\nBusiness Problem")
print("The telecom company is experiencing customer churn.")
print("The company wants to identify customers who are likely to leave and take preventive actions.")

print("\nBusiness Objective")
print("Reduce customer churn and improve customer retention.")

print("\nAI Solution")
print("Build an AI model that predicts whether a customer will churn.")
print("The prediction can help the company contact high-risk customers before they leave.")

print("\nInput Features")

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

print("\nExpected AI Output")

outputs = [
    "Customer Churn Prediction",
    "Churn Probability",
    "Risk Category",
    "Retention Recommendation"
]

for output in outputs:
    print("-", output)

print("\nRecommended AI Approach")

approach = [
    "Supervised Machine Learning",
    "Binary Classification",
    "Random Forest",
    "Logistic Regression"
]

for item in approach:
    print("-", item)

print("\nPROJECT 9 COMPLETED SUCCESSFULLY")



import pandas as pd

print("=" * 70)
print("PROJECT 10 : END-TO-END ML WORKFLOW")
print("=" * 70)

df = pd.read_excel("Telco_customer_churn.xlsx")

print("\nDataset Shape :", df.shape)

print("\nSTEP 1 : Business Problem")
print("Predict customer churn to improve retention.")

print("\nSTEP 2 : Data Collection")
print("Dataset :", len(df), "records")
print("Features :", df.shape[1])

print("\nSTEP 3 : Data Preparation")

print("Missing Values :", df.isnull().sum().sum())
print("Duplicate Records :", df.duplicated().sum())

print("\nSTEP 4 : Exploratory Data Analysis")

print("Average Monthly Charges :", round(df["Monthly Charges"].mean(),2))
print("Average Tenure :", round(df["Tenure Months"].mean(),2))
print("Average CLTV :", round(df["CLTV"].mean(),2))

print("\nCustomer Churn")

print(df["Churn Label"].value_counts())

print("\nSTEP 5 : Feature Selection")

selected_features = [
    "Tenure Months",
    "Monthly Charges",
    "Contract",
    "Internet Service",
    "Payment Method",
    "CLTV",
    "Churn Score"
]

for feature in selected_features:
    print("-", feature)

print("\nTarget Variable")

print("Churn Label")

print("\nSTEP 6 : Machine Learning Models")

models = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting"
]

for model in models:
    print("-", model)

print("\nSTEP 7 : Model Evaluation")

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]

for metric in metrics:
    print("-", metric)

print("\nSTEP 8 : Deployment")

deployment = [
    "Deploy using Streamlit",
    "Predict churn for new customers",
    "Display churn probability",
    "Generate retention recommendations"
]

for item in deployment:
    print("-", item)

print("\nSTEP 9 : Monitoring")

monitoring = [
    "Track prediction accuracy",
    "Monitor model performance",
    "Retrain model with new data",
    "Update business recommendations"
]

for item in monitoring:
    print("-", item)

print("\nSTEP 10 : Ethics")

ethics = [
    "Protect customer privacy",
    "Avoid discrimination",
    "Ensure transparency",
    "Use customer data responsibly"
]

for item in ethics:
    print("-", item)

print("\nML Workflow")

workflow = [
    "Business Problem",
    "Data Collection",
    "Data Preparation",
    "EDA",
    "Feature Engineering",
    "Model Training",
    "Model Evaluation",
    "Deployment",
    "Monitoring",
    "Business Decision"
]

for i, step in enumerate(workflow, 1):
    print(f"{i}. {step}")

print("\nPROJECT 10 COMPLETED SUCCESSFULLY")