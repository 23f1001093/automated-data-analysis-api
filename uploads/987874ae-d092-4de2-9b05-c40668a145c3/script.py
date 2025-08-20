import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Load the dataframe.
df = pd.read_csv('sample-weather.csv')

# Convert 'date' column to datetime objects
df['date'] = pd.to_datetime(df['date'])

# 1. Average temperature in Celsius
avg_temp = df['temperature_c'].mean()

# 2. Date of highest precipitation
max_precip_date = df.loc[df['precip_mm'].idxmax(), 'date'].strftime('%Y-%m-%d')

# 3. Minimum temperature
min_temp = df['temperature_c'].min()

# 4. Correlation between temperature and precipitation
correlation = df['temperature_c'].corr(df['precip_mm'])

# 5. Average precipitation in millimeters
avg_precip = df['precip_mm'].mean()

# 6. Temperature line chart
plt.figure(figsize=(10, 5))
plt.plot(df['date'], df['temperature_c'], color='red')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Over Time')
plt.tight_layout()
temp_buffer = BytesIO()
plt.savefig(temp_buffer, format='png')
temp_b64 = base64.b64encode(temp_buffer.getvalue()).decode('utf-8')
plt.close()


# 7. Precipitation histogram
plt.figure(figsize=(10, 5))
plt.hist(df['precip_mm'], color='orange')
plt.xlabel('Precipitation (mm)')
plt.ylabel('Frequency')
plt.title('Precipitation Distribution')
plt.tight_layout()
precip_buffer = BytesIO()
plt.savefig(precip_buffer, format='png')
precip_b64 = base64.b64encode(precip_buffer.getvalue()).decode('utf-8')
plt.close()

# Create the dictionary of results and save to result.json
results = {
    'average_temp_c': float(avg_temp),
    'max_precip_date': max_precip_date,
    'min_temp_c': int(min_temp),
    'temp_precip_correlation': float(correlation),
    'average_precip_mm': float(avg_precip),
    'temp_line_chart': temp_b64,
    'precip_histogram': precip_b64
}

import json
with open('result.json', 'w') as f:
    json.dump(results, f)