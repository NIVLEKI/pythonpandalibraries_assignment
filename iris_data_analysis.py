# iris_data_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# 1: Load and Explore the Dataset
try:
    iris = load_iris(as_frame=True)
    df = iris.frame
    print("Dataset loaded successfully!\n")
except Exception as e:
    print("Error loading dataset:", e)

# Display first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Check data types and missing values
print("\nData Types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

# No missing values; if any, we could handle like this:
# df.fillna(df.mean(), inplace=True)

# Task 2: Basic Data Analysis
print("\nBasic Statistics:")
print(df.describe())

# Group by species and get mean of each numeric feature
grouped = df.groupby("target").mean()
print("\nMean of each feature grouped by species (target):")
print(grouped)

# Rename target to species names
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))

# Task 3: Data Visualization

# 1. Line chart (Mean petal length per sample index for each species)
plt.figure(figsize=(10, 5))
for species in df["species"].unique():
    subset = df[df["species"] == species]
    plt.plot(subset.index, subset["petal length (cm)"], label=species)
plt.title("Petal Length Trend by Species")
plt.xlabel("Sample Index")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Bar chart (Average petal length per species)
plt.figure(figsize=(6, 4))
sns.barplot(x="species", y="petal length (cm)", data=df, palette="pastel")
plt.title("Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.show()

# 3. Histogram (Distribution of sepal length)
plt.figure(figsize=(6, 4))
plt.hist(df["sepal length (cm)"], bins=15, color='skyblue', edgecolor='black')
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 4. Scatter plot (Sepal length vs. Petal length)
plt.figure(figsize=(6, 4))
sns.scatterplot(x="sepal length (cm)", y="petal length (cm)", hue="species", data=df, palette="deep")
plt.title("Sepal Length vs. Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.show()

# Findings
print("\nObservations:")
print("- Setosa has significantly smaller petal length compared to other species.")
print("- Sepal length and petal length are positively correlated.")
print("- Petal length varies more between species than sepal width or length.")
