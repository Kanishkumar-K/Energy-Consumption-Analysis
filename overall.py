import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


# Load the data
data = pd.read_csv("data/data.csv")


# Define valid columns
valid_columns = ["Efficiency", "Appliance"]


# Check if valid columns exist in the dataset
missing_columns = [column for column in valid_columns if column not in data.columns]
if missing_columns:
    raise ValueError(f"Columns {missing_columns} not found in the dataset.")


# Select only valid columns
data = data[valid_columns]


# Drop rows with missing values if any
data = data.dropna()


# Encode categorical variables using LabelEncoder
label_encoder = LabelEncoder()
categorical_columns = ["Efficiency", "Appliance"]
for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])


# Bivariate Analysis: Scatter plot of Efficiency vs. Appliance
plt.scatter(data["Efficiency"], data["Appliance"])
plt.xlabel("Efficiency")
plt.ylabel("Appliance")
plt.title("Bivariate Analysis: Efficiency vs. Appliance")
plt.savefig("templates/bivariate.png")  # Save the plot as an image


# Univariate Analysis: Histogram of Appliance
plt.hist(data["Appliance"])
plt.xlabel("Appliance")
plt.ylabel("Frequency")
plt.title("Univariate Analysis: Appliance")
plt.savefig("templates/univariate.png")  # Save the plot as an image


# Multivariate Analysis: Correlation Matrix Heatmap
correlation_matrix = data.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="RdYlGn")
plt.title("Multivariate Analysis: Correlation Matrix")
plt.savefig("templates/multivariate.png")  # Save the plot as an image


# Random Forest Analysis
X = data.drop(["Appliance"], axis=1)  # Features
y = data["Appliance"]  # Target variable


# Create and fit the random forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)


# Feature Importance
importance = rf.feature_importances_
feature_names = X.columns


# Plotting feature importances
plt.barh(range(len(importance)), importance, align="center")
plt.yticks(range(len(feature_names)), feature_names)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Random Forest: Feature Importance")
plt.savefig("templates/feature_importance.png")  # Save the plot as an image


# Generate HTML file (same code as before)
html_content = f'''
<html>
<head>
    <title>Data Analysis Plots</title>
</head>
<body>
    <h1>Bivariate Analysis: Efficiency vs. Appliance</h1>
    <img src="templates/bivariate.png" alt="Bivariate Analysis">


    <h1>Univariate Analysis: Appliance</h1>
    <img src="templates/univariate.png" alt="Univariate Analysis">


    <h1>Multivariate Analysis: Correlation Matrix</h1>
    <img src="templates/multivariate.png" alt="Multivariate Analysis">


    <h1>Random Forest: Feature Importance</h1>
    <img src="templates/feature_importance.png" alt="Feature Importance">
</body>
</html>
'''


# Save HTML file (same code as before)
with open("templates/analysis.html", "w") as file:
    file.write(html_content)


