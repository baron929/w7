import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Display the first few rows of the dataset
print(df.head())

# Check the data types of each column
print(df.dtypes)

# Check for missing values
print(df.isnull().sum())

# Compute basic statistics of the numerical columns
print(df.describe())# Perform groupings on the 'species' column and compute the mean of 'sepal length (cm)' for each group
print(df.groupby('species')['sepal length (cm)'].mean())

import matplotlib.pyplot as plt

# Create a line chart showing the trend of sepal length over time (not applicable for Iris dataset)
# Instead, we can create a line chart showing the trend of sepal length for each species
plt.figure(figsize=(10,6))
for species in df['species'].unique():
    species_df = df[df['species'] == species]
    plt.plot(species_df['sepal length (cm)'], label=species)

plt.title('Sepal Length Trend for Each Species')
plt.xlabel('Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.show()
# Create a bar chart showing the comparison of average petal length per species
plt.figure(figsize=(10,6))
plt.bar(df['species'].unique(), df.groupby('species')['petal length (cm)'].mean())
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.show()# Create a histogram of sepal length to understand its distribution
plt.figure(figsize=(10,6))
plt.hist(df['sepal length (cm)'], bins=10)
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()
# Create a scatter plot to visualize the relationship between sepal length and petal length
plt.figure(figsize=(10,6))
plt.scatter(df['sepal length (cm)'], df['petal length (cm)'])
plt.title('Relationship between Sepal Length and Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.show()
