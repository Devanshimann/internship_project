import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
print("CAPSTONE PROJECT 3 : SALES / DEMAND FORECASTING")


df = pd.read_csv("Walmart.csv")

print("\nDataset Shape :", df.shape)

df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")

sales = (
    df.groupby("Date")["Weekly_Sales"]
    .sum()
    .reset_index()
)

print("\nFirst Five Rows")
print(sales.head())

sales["Month"] = sales["Date"].dt.month_name()

monthly_sales = (
    sales.groupby("Month")["Weekly_Sales"]
    .sum()
    .reset_index()
)

print("\nMonthly Sales")
print(monthly_sales)

plt.figure(figsize=(12,5))
plt.plot(sales["Date"], sales["Weekly_Sales"])
plt.title("Weekly Sales Trend")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.show()

holiday_sales = (
    df.groupby("Holiday_Flag")["Weekly_Sales"]
    .mean()
    .reset_index()
)

print("\nAverage Sales During Holidays")
print(holiday_sales)

sales["Week"] = np.arange(len(sales))

X = sales[["Week"]]
y = sales["Weekly_Sales"]

model = LinearRegression()

model.fit(X, y)

sales["Predicted Sales"] = model.predict(X)

future = pd.DataFrame({
    "Week": np.arange(len(sales), len(sales)+12)
})

future["Forecast Sales"] = model.predict(future)

print("\nNext 12 Week Forecast")

print(future)

plt.figure(figsize=(12,5))

plt.plot(
    sales["Week"],
    sales["Weekly_Sales"],
    label="Actual Sales"
)

plt.plot(
    sales["Week"],
    sales["Predicted Sales"],
    label="Predicted Sales"
)

plt.plot(
    future["Week"],
    future["Forecast Sales"],
    "--",
    label="Forecast"
)

plt.legend()

plt.title("Sales Forecast")

plt.show()

mae = mean_absolute_error(
    y,
    sales["Predicted Sales"]
)

rmse = np.sqrt(
    mean_squared_error(
        y,
        sales["Predicted Sales"]
    )
)

print("\nModel Performance")

print("MAE :", round(mae,2))
print("RMSE :", round(rmse,2))


future.to_csv(
    "Sales_Forecast.csv",
    index=False
)

print("\nForecast file saved successfully.")

print("\nCAPSTONE PROJECT 3 COMPLETED SUCCESSFULLY")