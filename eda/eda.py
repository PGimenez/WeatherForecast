import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure the 'plots' directory exists
if not os.path.exists('data/plots'):
    os.makedirs('data/plots')

# Load the dataset
csv_file = "data/all_cities.csv"
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file, parse_dates=['date'])
else:
    raise FileNotFoundError(f"{csv_file} does not exist.")

# Data Overview

# Display first few rows
print("First five rows of the dataset:")
print(df.head())

# Dataset shape
print(f"\nDataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

# Column data types
print("\nData types of each column:")
print(df.dtypes)

# Unique cities
print("\nList of cities in the dataset:")
print(df['city'].unique())

# Handling Missing Values

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values in each column:")
print(missing_values[missing_values > 0])

# Optionally, fill missing values or drop them
df.dropna(inplace=True)

# Statistical Summaries

# Describe numerical columns
print("\nStatistical summary of numerical columns:")
print(df.describe())

# Describe categorical columns
print("\nStatistical summary of categorical columns:")
print(df.describe(include=['object', 'category']))

# Visualizations

# 1. Temperature over Time for Each City
plt.figure(figsize=(15, 8))
for city in df['city'].unique():
    city_data = df[df['city'] == city]
    plt.plot(city_data['date'], city_data['temperature_2m'], label=city)
plt.xlabel('Date')
plt.ylabel('Temperature at 2m (°C)')
plt.title('Temperature Over Time for Each City')
plt.legend()
plt.savefig('data/plots/temperature_over_time.png')
plt.close()

# 2. Average Temperature by City
avg_temp_by_city = df.groupby('city')['temperature_2m'].mean().reset_index()
plt.figure(figsize=(12, 6))
sns.barplot(data=avg_temp_by_city, x='city', y='temperature_2m')
plt.xticks(rotation=45)
plt.xlabel('City')
plt.ylabel('Average Temperature at 2m (°C)')
plt.title('Average Temperature by City')
plt.tight_layout()
plt.savefig('data/plots/average_temperature_by_city.png')
plt.close()

# 3. Distribution of Humidity
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='relative_humidity_2m', bins=30, kde=True)
plt.xlabel('Relative Humidity at 2m (%)')
plt.title('Distribution of Relative Humidity')
plt.savefig('data/plots/humidity_distribution.png')
plt.close()

# 4. Correlation Heatmap
variables_of_interest = [
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
    'apparent_temperature', 'precipitation', 'wind_speed_10m',
    'pressure_msl', 'cloud_cover'
]
corr_matrix = df[variables_of_interest].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap of Selected Variables')
plt.tight_layout()
plt.savefig('data/plots/correlation_heatmap.png')
plt.close()

# 5. Boxplot of Temperature by City
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='city', y='temperature_2m')
plt.xticks(rotation=45)
plt.xlabel('City')
plt.ylabel('Temperature at 2m (°C)')
plt.title('Temperature Distribution by City')
plt.tight_layout()
plt.savefig('data/plots/temperature_boxplot_by_city.png')
plt.close()

# 6. Wind Speed Distribution
plt.figure(figsize=(12, 6))
for city in df['city'].unique():
    sns.kdeplot(data=df[df['city'] == city], x='wind_speed_10m', label=city, shade=True)
plt.xlabel('Wind Speed at 10m (m/s)')
plt.title('Wind Speed Distribution by City')
plt.legend()
plt.savefig('data/plots/wind_speed_distribution.png')
plt.close()

# 7. Time Series of Precipitation
plt.figure(figsize=(15, 8))
for city in df['city'].unique():
    city_data = df[df['city'] == city]
    plt.plot(city_data['date'], city_data['precipitation'].cumsum(), label=city)
plt.xlabel('Date')
plt.ylabel('Cumulative Precipitation (mm)')
plt.title('Cumulative Precipitation Over Time')
plt.legend()
plt.savefig('data/plots/cumulative_precipitation.png')
plt.close()

# 8. Scatter Plot of Temperature vs. Humidity
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='temperature_2m', y='relative_humidity_2m', hue='city', s=50)
plt.xlabel('Temperature at 2m (°C)')
plt.ylabel('Relative Humidity at 2m (%)')
plt.title('Temperature vs. Relative Humidity')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('data/plots/temperature_vs_humidity.png')
plt.close()

# 9. Monthly Average Temperature
df['month'] = df['date'].dt.to_period('M')
monthly_avg_temp = df.groupby(['month', 'city'])['temperature_2m'].mean().reset_index()
monthly_avg_temp['month'] = monthly_avg_temp['month'].astype(str)
plt.figure(figsize=(15, 8))
sns.lineplot(data=monthly_avg_temp, x='month', y='temperature_2m', hue='city', marker='o')
plt.xlabel('Month')
plt.ylabel('Average Temperature at 2m (°C)')
plt.title('Monthly Average Temperature by City')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('data/plots/monthly_average_temperature.png')
plt.close()

# 10. Wind Rose Plot (Optional)
# Note: Requires windrose library
# Install with: pip install windrose
from windrose import WindroseAxes

def plot_wind_rose(city_name):
    city_data = df[df['city'] == city_name]
    wind_direction = city_data['wind_direction_10m']
    wind_speed = city_data['wind_speed_10m']

    ax = WindroseAxes.from_ax()
    ax.bar(wind_direction, wind_speed, normed=True, opening=0.8, edgecolor='white')
    ax.set_title(f'Wind Rose for {city_name}')
    ax.set_legend()
    plt.savefig('data/plots/wind_rose_{city_name}.png')
    plt.close()

# Plot wind rose for each city
for city in df['city'].unique():
    plot_wind_rose(city)
