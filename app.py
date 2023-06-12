from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Home route
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract user input from the form
    power_rating = float(request.form['power_rating'])
    usage_pattern = float(request.form['usage_pattern'])
    energy_efficiency_rating = int(request.form['energy_efficiency_rating'])

    # Generate energy consumption prediction and plot
    energy_consumption = predict_energy_consumption(power_rating, usage_pattern, energy_efficiency_rating)
    plot_path = visualize_energy_consumption(power_rating, usage_pattern, energy_efficiency_rating)

    # Generate suggestions using OpenAI based on the plot
    suggestions = generate_suggestions(power_rating, usage_pattern, energy_efficiency_rating)

    # Render the result page with energy consumption, plot, and suggestions
    return render_template('result.html', energy_consumption=energy_consumption, plot_path=plot_path, suggestions=suggestions)


# Function to predict energy consumption
def predict_energy_consumption(power_rating, usage_pattern, energy_efficiency_rating):
    data = pd.read_csv('data/appliance.csv')
    X = data[['power_rating', 'usage_pattern', 'energy_efficiency_rating']]
    y = data['energy_consumption']

    model = LinearRegression()
    model.fit(X, y)

    input_data = pd.DataFrame({'power_rating': [power_rating], 'usage_pattern': [usage_pattern], 'energy_efficiency_rating': [energy_efficiency_rating]})
    prediction = model.predict(input_data)

    return prediction[0]

# Future trends route
@app.route('/future-trends')
def future_trends():
    data = pd.read_csv('data/appliance.csv')
    x = data['power_rating']
    y = data['energy_consumption']

    # Perform linear regression
    model = LinearRegression()
    model.fit(x.values.reshape(-1, 1), y)

    # Predict future trends
    future_x = np.linspace(min(x), max(x), num=100)
    future_y = model.predict(future_x.reshape(-1, 1))

    # Plot the future trends
    plt.figure()
    plt.plot(x, y, label='Actual')
    plt.plot(future_x, future_y, color='green', linestyle='--', label='Future Trend')
    plt.xlabel('Power Rating')
    plt.ylabel('Energy Consumption')
    plt.title('Future Energy Consumption Trends')
    plt.legend()

    # Save the plot
    plot_path1 = 'static/future_trends.png'
    plt.savefig(plot_path1)
    plt.close()

    return render_template('future_trends.html', plot_path1=plot_path1)
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the data points and linear regression
    ax1.scatter(x, y, label='Actual')
    ax1.scatter(power_rating, predict_energy_consumption(power_rating, usage_pattern, energy_efficiency_rating),
                color='red', label='User Input')
    model = LinearRegression()
    model.fit(x.values.reshape(-1, 1), y)
    ax1.plot(x, model.predict(x.values.reshape(-1, 1)), color='orange', label='Linear Regression')
    ax1.set_xlabel('Power Rating')
    ax1.set_ylabel('Energy Consumption')
    ax1.set_title('Energy Consumption Analysis')
    ax1.legend()

    # Plot the energy consumption trend
    trend_data = pd.read_csv('data/trend.csv')
    trend_x = trend_data['day']
    trend_y = trend_data['energy_consumption']
    ax2.plot(trend_x, trend_y, marker='o')
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Energy Consumption')
    ax2.set_title('Energy Consumption Trend')
    ax2.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot
    plot_path = 'static/plot.png'
    plt.savefig(plot_path)
    plt.close()

    return plot_path

if __name__ == '__main__':
    app.run(debug=True)
