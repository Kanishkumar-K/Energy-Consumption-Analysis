import pandas as pd

# Load the dataset
data = pd.read_csv('data/data.csv')

# Calculate descriptive statistics
statistics = data.describe()

# Print the descriptive statistics
print(statistics)
