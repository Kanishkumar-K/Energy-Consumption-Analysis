import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.DataFrame({
    'power_rating': [100, 200, 150, 250],
    'usage_pattern': [1.5, 2.0, 1.2, 2.5],
    'energy_efficiency_rating': [0.8, 0.9, 0.85, 0.95],
    'energy_consumption': [120, 180, 150, 220]
})

# Function to predict energy consumption
def predict_energy_consumption(power_rating, usage_pattern, energy_efficiency_rating):
    X = data[['power_rating', 'usage_pattern', 'energy_efficiency_rating']]
    y = data['energy_consumption']

    model = LinearRegression()
    model.fit(X, y)

    input_data = pd.DataFrame({'power_rating': [power_rating], 'usage_pattern': [usage_pattern], 'energy_efficiency_rating': [energy_efficiency_rating]})
    prediction = model.predict(input_data)

    return prediction[0]

# Function to generate suggestions
def generate_suggestions(power_rating, usage_pattern, energy_efficiency_rating):
    suggestions = []

    # Check power rating
    if power_rating > 200:
        suggestions.append("Consider using appliances with lower power ratings.")
    else:
        suggestions.append("Ensure your appliances are energy-efficient.")

    # Check usage pattern
    if usage_pattern < 1.0:
        suggestions.append("Optimize your usage pattern to avoid unnecessary energy consumption.")
    elif usage_pattern > 2.0:
        suggestions.append("Adjust your usage pattern to reduce energy consumption during peak hours.")
    
    # Check energy efficiency rating
    if energy_efficiency_rating == 1:
        suggestions.append("Upgrade to appliances with higher energy efficiency ratings.")
    elif energy_efficiency_rating == 2:
        suggestions.append("Ensure regular maintenance of your appliances to maximize energy efficiency.")
    
    return suggestions

# Function to visualize energy consumption
def visualize_energy_consumption(power_rating, usage_pattern, energy_efficiency_rating):
    data = pd.read_csv('data/appliance.csv')
    x = data['power_rating']
    y = data['energy_consumption']

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Plot the data points and linear regression
    axs[0].scatter(x, y, label='Actual')
    axs[0].scatter(power_rating, predict_energy_consumption(power_rating, usage_pattern, energy_efficiency_rating),
                   color='red', label='User Input')
    model = LinearRegression()
    model.fit(x.values.reshape(-1, 1), y)
    axs[0].plot(x, model.predict(x.values.reshape(-1, 1)), color='orange', label='Linear Regression')
    axs[0].set_xlabel('Power Rating')
    axs[0].set_ylabel('Energy Consumption')
    axs[0].set_title('Energy Consumption Analysis')
    axs[0].legend()

    # Plot the energy consumption trend
    trend_data = pd.read_csv('data/trend.csv')
    trend_x = trend_data['day']
    trend_y = trend_data['energy_consumption']
    axs[1].plot(trend_x, trend_y, marker='o')
    axs[1].set_xlabel('Day')
    axs[1].set_ylabel('Energy Consumption')
    axs[1].set_title('Energy Consumption Trend')
    axs[1].grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot
    plot_path = 'static/plot.png'
    plt.savefig(plot_path)
    plt.close()

    return plot_path


