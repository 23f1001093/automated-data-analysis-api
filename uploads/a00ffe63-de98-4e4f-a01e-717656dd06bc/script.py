import pandas as pd
import matplotlib.pyplot as plt
import base64
import json

df = pd.read_csv('sample-weather.csv')

# Convert date to datetime objects
df['date'] = pd.to_datetime(df['date'])

# 1. Average temperature in Celsius
avg_temp_c = df['temperature_c'].mean()

# 2. Date of highest precipitation
max_precip_date = df.loc[df['precipitation_mm'].idxmax(), 'date'].strftime('%Y-%m-%d')

# 3. Minimum temperature recorded
min_temp_c = df['temperature_c'].min()

# 4. Correlation between temperature and precipitation
temp_precip_correlation = df['temperature_c'].corr(df['precipitation_mm'])

# 5. Average precipitation in millimeters
avg_precip_mm = df['precipitation_mm'].mean()

# 6. Temperature over time line chart
plt.figure(figsize=(10, 5))
plt.plot(df['date'], df['temperature_c'], color='red')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Temperature over Time')
plt.xticks(rotation=45)
plt.tight_layout()
temp_chart = base64.b64encode(plt.savefig(None, format='png')).decode('utf-8')
plt.close()

# 7. Precipitation histogram
plt.figure(figsize=(8, 6))
plt.hist(df['precipitation_mm'], color='orange', bins=20)
plt.xlabel('Precipitation (mm)')
plt.ylabel('Frequency')
plt.title('Precipitation Distribution')
plt.tight_layout()
precip_hist = base64.b64encode(plt.savefig(None, format='png')).decode('utf-8')
plt.close()

# Create and save the JSON object
result = {
    "average_temp_c": avg_temp_c,
    "max_precip_date": max_precip_date,
    "min_temp_c": min_temp_c,
    "temp_precip_correlation": temp_precip_correlation,
    "average_precip_mm": avg_precip_mm,
    "temp_line_chart": temp_chart,
    "precip_histogram": precip_hist
}

with open('result.json', 'w') as f:
    json.dump(result, f, indent=4)
