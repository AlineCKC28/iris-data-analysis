# ==============================
# Task: Load, Explore, Analyze, and Visualize a Dataset
# ==============================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# ------------------------------
# Task 1: Load and Explore the Dataset
# ------------------------------

try:
    # Load Iris dataset from sklearn
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    
    print("First 5 rows of the dataset:")
    print(data.head())

    print("\nDataset info:")
    print(data.info())

    print("\nMissing values per column:")
    print(data.isnull().sum())

    # If there were missing values, we could fill or drop them
    # data = data.fillna(method='ffill')  # fill forward as an example
    # data = data.dropna()  # or drop rows with missing values

except FileNotFoundError:
    print("Error: Dataset file not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# ------------------------------
# Task 2: Basic Data Analysis
# ------------------------------

print("\nBasic statistics for numerical columns:")
print(data.describe())

# Group by species and compute mean for each numeric column
print("\nMean values by species:")
print(data.groupby('species').mean())

# Example: Identify patterns
print("\nObservations:")
print("- Setosa species tends to have smaller petal length and width than Versicolor and Virginica.")
print("- Sepal length is relatively similar between Versicolor and Virginica.")

# ------------------------------
# Task 3: Data Visualization
# ------------------------------

sns.set(style="whitegrid")  # improve plot style

# 1. Line chart (mean sepal length by species)
plt.figure(figsize=(8,5))
data.groupby('species')['sepal length (cm)'].mean().plot(kind='line', marker='o')
plt.title("Average Sepal Length by Species (Line Chart)")
plt.xlabel("Species")
plt.ylabel("Average Sepal Length (cm)")
plt.show()

# 2. Bar chart (average petal length per species)
plt.figure(figsize=(8,5))
sns.barplot(x='species', y='petal length (cm)', data=data, palette='pastel')
plt.title("Average Petal Length by Species (Bar Chart)")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# 3. Histogram (distribution of sepal width)
plt.figure(figsize=(8,5))
plt.hist(data['sepal width (cm)'], bins=10, color='skyblue', edgecolor='black')
plt.title("Distribution of Sepal Width (Histogram)")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter plot (sepal length vs petal length)
plt.figure(figsize=(8,5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=data, palette='deep')
plt.title("Sepal Length vs Petal Length (Scatter Plot)")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title='Species')
plt.show()
