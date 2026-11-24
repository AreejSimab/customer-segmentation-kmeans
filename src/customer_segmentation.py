"""Customer segmentation with K-Means (Task 3)

Run from the repository root:  python src/customer_segmentation.py
The raw data must be in data/ (see data/README.md).
"""
import os


# Objective
# ------------------------------------------------------------

# 1. Import Libraries
# ------------------------------------------------------------
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np


# 2. Load the Data
# ------------------------------------------------------------
df = pd.read_csv("data/Mall_Customers.csv")

print(df.shape)
df.head(3)


# 3. Check the Data
# ------------------------------------------------------------
df.info()

df.isnull().sum()

print("Duplicated Rows:",df.duplicated().sum())

df["Gender"].value_counts()

df.drop(columns="CustomerID").describe()


# 4. Clean the Columns
# ------------------------------------------------------------
df = df.drop(columns="CustomerID")
df = df.rename(columns={"Annual Income (k$)":"Income",
          "Spending Score (1-100)":"Spending"})

df.head()


# 5. Exploratory Data Analysis (EDA)
# ------------------------------------------------------------
plt.figure(figsize=(12,3.5))

plt.subplot(1,3,1)
sns.histplot(df["Age"],bins=15)
plt.title("Age")

plt.subplot(1,3,2)
sns.histplot(df["Income"],bins=15)
plt.title("Annual Income(k$)")

plt.subplot(1,3,3)
sns.histplot(df["Spending"],bins=15)
plt.title("Spending Score")

plt.tight_layout()
plt.show()  # close the window to continue



# 5.2 Relationships between columns
# ------------------------------------------------------------
plt.figure(figsize=(11, 4.5))
plt.subplot(1,2,1)
sns.heatmap(df[["Age","Income","Spending"]].corr(),annot=True)
plt.title("Correlation")

plt.subplot(1,2,2)
sns.scatterplot(x="Income",y="Spending",data=df)
plt.title("Income and Spending")

plt.tight_layout
plt.show()  # close the window to continue


# 6. Feature Scaling
# ------------------------------------------------------------
from sklearn.preprocessing import StandardScaler

x = df[["Income","Spending"]]

sc = StandardScaler()
x_scaled = sc.fit_transform(x)

#Checking the scaled column has mean 0 and std 1
print(x_scaled.mean(axis=0).round(2))
print(x_scaled.std(axis=0).round(2))


# 7. Choosing the Number of Clusters (K)
# ------------------------------------------------------------
from sklearn.cluster import KMeans

inertia = []
for k in range(1,11):
    km = KMeans(n_clusters=k,n_init=10,random_state=42)
    km.fit(x_scaled)
    inertia.append(km.inertia_)
print([round(i, 1) for i in inertia])


plt.figure(figsize=(7, 4))
sns.lineplot(x=range(1, 11), y=inertia, marker="o")
plt.xticks(range(1, 11))
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow method")
plt.grid(axis = "x")
plt.show()  # close the window to continue


# 7.2 Silhouette score
# ------------------------------------------------------------
from sklearn.metrics import silhouette_score

ss = []

for k in range(2,11):
    km1 = KMeans(n_clusters=k,n_init=10,random_state=42)
    labels = km1.fit_predict(x_scaled)
    ss.append(silhouette_score(x_scaled,labels))

for k, s in zip(range(2,11),ss):
    print(k,round(s,4))

plt.figure(figsize=(7,4))
sns.lineplot(x=range(2,11),y = ss, marker="o")
plt.xticks(range(2,11))
plt.xlabel("Number of clusters (K)")
plt.ylabel("Silhouette score")
plt.title("Silhouette score by K")
plt.show()  # close the window to continue



# 8. Fitting K-Means with K = 5
# ------------------------------------------------------------
kmeans = KMeans(n_clusters=5,n_init=10,random_state=42)
df["Cluster"] = kmeans.fit_predict(x_scaled)


print(round(silhouette_score(x_scaled,df["Cluster"]),4))
print(df["Cluster"].value_counts().sort_index())
df.head()


# 9. Visualizing the Clusters
# ------------------------------------------------------------
centers = sc.inverse_transform(kmeans.cluster_centers_)

plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x="Income", y="Spending", hue="Cluster")
plt.scatter(centers[:,0],centers[:,1],c="black",s=250,marker="X",label="Centroids")
plt.xlabel("Annual income (k$)")
plt.ylabel("Spending score (1-100)")
plt.title("Customer segments")
plt.legend()
plt.show()  # close the window to continue




# 10. Profiling the Segments
# ------------------------------------------------------------
profile = df.groupby("Cluster")[["Age","Income","Spending"]].mean()
profile["Customers"]= df["Cluster"].value_counts().sort_index()
profile["Female %"] = (df["Gender"].eq("Female").groupby(df["Cluster"]).mean() * 100).round(0)
print(profile)


# 11. Segment Names and Recommendations
# ------------------------------------------------------------

# 12. Conclusion
# ------------------------------------------------------------