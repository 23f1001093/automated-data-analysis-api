import pandas as pd
import json
import matplotlib.pyplot as plt
import base64
import io

# Load the dataframe.
df = pd.read_csv('sample-weather.csv')

# Get the actual column names
temp_col = df.columns[1]
precip_col = df.columns[2]
date_col = df.columns[0]

# 1. Average temperature
avg_temp = df[temp_col].mean()

# 2. Date of highest precipitation
max_precip_date = df.loc[df[precip_col].idxmax(), date_col]

# 3. Minimum temperature
min_temp = df[temp_col].min()

# 4. Correlation
correlation = df[temp_col].corr(df[precip_col])

# 5. Average precipitation
avg_precip = df[precip_col].mean()

# 6. Temperature line chart
plt.figure(figsize=(10, 5))
plt.plot(df[date_col], df[temp_col], color='red')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Temperature over Time')
plt.xticks(rotation=45)
plt.tight_layout()
temp_chart = io.BytesIO()
plt.savefig(temp_chart, format='png')
temp_chart_b64 = base64.b64encode(temp_chart.getvalue()).decode('utf-8')

# 7. Precipitation histogram
plt.figure(figsize=(10, 5))
plt.hist(df[precip_col], color='orange')
plt.xlabel('Precipitation (mm)')
plt.ylabel('Frequency')
plt.title('Precipitation Distribution')
plt.tight_layout()
precip_hist = io.BytesIO()
plt.savefig(precip_hist, format='png')
precip_hist_b64 = base64.b64encode(precip_hist.getvalue()).decode('utf-8')
plt.close('all')

# Create and save the result dictionary.
result = {
    "average_temp_c": float(avg_temp),
    "max_precip_date": str(max_precip_date),
    "min_temp_c": float(min_temp),
    "temp_precip_correlation": float(correlation),
    "average_precip_mm": float(avg_precip),
    "temp_line_chart": temp_chart_b64,
    "precip_histogram": precip_hist_b64
}

with open('result.json', 'w') as f:
    json.dump(result, f)