import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 70)
print("PROJECT 8 : CUSTOMER SEGMENTATION")
print("=" * 70)

df = pd.read_csv("Mall_Customers.csv")

print("\nDataset Shape :", df.shape)

print("\nFirst Five Rows")

print(df.head())

print("\nDataset Information")

print(df.info())

print("\nSummary Statistics")

print(df.describe())

X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

wcss = []

for i in range(1,11):

    model = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    wcss.append(model.inertia_)

plt.figure(figsize=(7,5))

plt.plot(range(1,11),wcss,marker="o")

plt.xlabel("Clusters")

plt.ylabel("WCSS")

plt.title("Elbow Method")

plt.show()

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

score = silhouette_score(
    X_scaled,
    df["Cluster"]
)

print("\nSilhouette Score :",round(score,3))

print("\nCluster Distribution")

print(df["Cluster"].value_counts())

centers = scaler.inverse_transform(
    kmeans.cluster_centers_
)

centers = pd.DataFrame(
    centers,
    columns=[
        "Annual Income",
        "Spending Score"
    ]
)

print("\nCluster Centers")

print(centers)

plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="Set2",
    s=80
)

plt.title("Customer Segments")

plt.show()

print("\nBusiness Recommendations")

print("High income and high spending customers should receive loyalty rewards.")

print("High income but low spending customers should receive personalized offers.")

print("Low income customers can be targeted with affordable products.")

print("\n")
print("=" * 70)
print("CAPSTONE PROJECT 2 : CUSTOMER SEGMENTATION & PROFILING")
print("=" * 70)

profile = df.groupby("Cluster").agg({

    "CustomerID":"count",

    "Age":"mean",

    "Annual Income (k$)":"mean",

    "Spending Score (1-100)":"mean"

}).round(2)

profile.rename(columns={

    "CustomerID":"Customers"

},inplace=True)

print("\nCustomer Profiles")

print(profile)

plt.figure(figsize=(7,5))

sns.boxplot(
    data=df,
    x="Cluster",
    y="Age"
)

plt.title("Age by Cluster")

plt.show()

plt.figure(figsize=(7,5))

sns.boxplot(
    data=df,
    x="Cluster",
    y="Annual Income (k$)"
)

plt.title("Income by Cluster")

plt.show()

plt.figure(figsize=(7,5))

sns.boxplot(
    data=df,
    x="Cluster",
    y="Spending Score (1-100)"
)

plt.title("Spending Score by Cluster")

plt.show()

plt.figure(figsize=(7,5))

sns.countplot(
    data=df,
    x="Cluster",
    hue="Gender"
)

plt.title("Gender Distribution")

plt.show()

print("\nCustomer Profiles")

for cluster in sorted(df["Cluster"].unique()):

    temp = df[df["Cluster"]==cluster]

    print("\nCluster",cluster)

    print("Customers :",len(temp))

    print("Average Age :",round(temp["Age"].mean(),2))

    print("Average Income :",round(temp["Annual Income (k$)"].mean(),2))

    print("Average Spending :",round(temp["Spending Score (1-100)"].mean(),2))
