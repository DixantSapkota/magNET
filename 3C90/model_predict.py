import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import load_model

model = load_model('fft32_3C90.h5')

# Load the data from the CSV file
data = pd.read_csv('B_aggregate_validate_3C90.csv', encoding='utf-8')

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 5))

# Extract time series data and target variable
time_series_data = data.iloc[:, 1:-4].values  # Selecting the first 1024 columns
y = data.iloc[:, -2].values  # The last column (loss)
# Assuming your CSV file has columns 'Frequency' and 'Temperature'
frequency = data['Freq'].values
temperature = data['Temp'].values
classification = data['Classification'].values

# Normalize frequency and temperature (assuming they are 1D arrays)
frequency = scaler.fit_transform(frequency.reshape(-1, 1))
temperature = scaler.fit_transform(temperature.reshape(-1, 1))

# Reshape time series data for LSTM
time_series_data = time_series_data.reshape((-1, 32, 1))

# Normalize the time series data if needed
scaler = MinMaxScaler(feature_range=(0, 5))
time_series_data = scaler.fit_transform(time_series_data.reshape(-1, 1))
time_series_data = time_series_data.reshape((-1, 32, 1))

# Normalize the target variable (Area)
target_scaler = MinMaxScaler(feature_range=(.001, 4))
y = y.reshape(-1, 1)
y_test = target_scaler.fit_transform(y)

predictions = model.predict([time_series_data, frequency, temperature, classification])
predictions_original_scale = target_scaler.inverse_transform(predictions).ravel()

#----------------------------------------------------------------------------------
relative_error = (np.abs(y_test - predictions) / y_test) * 100
print (predictions_original_scale)
#----------------------------------------------------------------------------------

avg_error = np.mean(relative_error)
max_error = np.max(relative_error)
percentile_95 = np.percentile(relative_error, 95)

# Plot the histogram
plt.hist(relative_error, bins=30, edgecolor='k', alpha=0.7, density= True)
plt.xlabel('Relative Error (%)')
plt.ylabel('Density of Data Points')
title_names = "Histogram of Relative Error (3C90)"
title_values = f"Avg: {avg_error:.2f}%, 95-Prct: {percentile_95:.2f}%, Max: {max_error:.2f}%"
plt.title(f"{title_names}\n{title_values}", loc='center')

# Plot vertical lines for average, max and 95th percentile
plt.axvline(avg_error, color='r', linestyle='dashed', linewidth=1)
plt.axvline(max_error, color='g', linestyle='dotted', linewidth=1)
plt.axvline(percentile_95, color='b', linestyle='dotted', linewidth=1)

# Annotate the lines
plt.annotate('Avg: {:.2f}%'.format(avg_error), (avg_error, plt.gca().get_ylim()[1] * 0.9), textcoords="offset points", xytext=(0,10), ha='center')
plt.annotate('Max: {:.2f}%'.format(max_error), (max_error, plt.gca().get_ylim()[1] * 0.8), textcoords="offset points", xytext=(0,10), ha='center')
plt.annotate('95th Percentile: {:.2f}%'.format(percentile_95), (percentile_95, plt.gca().get_ylim()[1] * 0.7), textcoords="offset points", xytext=(0,10), ha='center')

plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('3C90_Histogram.png')
plt.show()
