import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
iris = sns.load_dataset('iris')

# Display first 5 rows
print("First 5 rows of the dataset:")
print(iris.head())

# Dataset info
print("\nDataset Info:")
iris.info()

# Summary statistics
print("\nStatistical Summary:")
print(iris.describe())

# Missing values
print("\nMissing Values:")
print(iris.isnull().sum())

# Duplicate rows
print("\nDuplicate Rows:")
print(iris.duplicated().sum())

# Scatter Plot 
plt.figure(figsize=(8, 6))
sns.scatterplot(data=iris, x='sepal_length', y='sepal_width', hue='species', palette='Set1')
plt.title('Sepal Length vs Sepal Width by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.grid(True)
plt.show()

# Histogram 
plt.figure(figsize=(8, 6))
sns.histplot(data=iris, x='petal_length', bins=20, hue='species', kde=True, multiple='stack', palette='pastel')
plt.title('Distribution of Petal Length by Species')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Count')
plt.grid(True)
plt.show()

#  Box Plot
plt.figure(figsize=(8, 6))
sns.boxplot(data=iris, x='species', y='petal_width', palette='Accent')
plt.title('Petal Width by Species')
plt.xlabel('Species')
plt.ylabel('Petal Width (cm)')
plt.grid(True)
plt.show()
