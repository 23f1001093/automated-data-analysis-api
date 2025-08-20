import pandas as pd
import matplotlib.pyplot as plt
import base64
import json
import io

# Load the dataframe.
df = pd.read_csv('sample-weather.csv')

# Initial data exploration
columns = df.columns.tolist()
head = df.head().to_string()
with open('metadata.txt', 'w') as f:
    f.write('Columns:\n' + str(columns) + '\n\nHead:\n' + head)

# 1. Average temperature
avg_temp = df['temperature_c'].mean()

# 2. Date of highest precipitation
max_precip_date = df.loc[df['precip_mm'].idxmax(), 'date']

# 3. Minimum temperature
min_temp = df['temperature_c'].min()

# 4. Correlation
correlation = df['temperature_c'].corr(df['precip_mm'])

# 5. Average precipitation
avg_precip = df['precip_mm'].mean()

# 6. Temperature line chart
plt.figure(figsize=(10, 5))
plt.plot(df['date'], df['temperature_c'], color='red')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Temperature over Time')
plt.xticks(rotation=45)
plt.tight_layout()
temp_buf = io.BytesIO()
plt.savefig(temp_buf, format='png')
temp_base64 = base64.b64encode(temp_buf.getvalue()).decode('utf-8')
plt.close()

# 7. Precipitation histogram
plt.figure(figsize=(10, 5))
plt.hist(df['precip_mm'], color='orange')
plt.xlabel('Precipitation (mm)')
plt.ylabel('Frequency')
plt.title('Precipitation Distribution')
plt.tight_layout()
precip_buf = io.BytesIO()
plt.savefig(precip_buf, format='png')
precip_base64 = base64.b64encode(precip_buf.getvalue()).decode('utf-8')
plt.close()

# Create the JSON object
result = {
    "average_temp_c": avg_temp,
    "max_precip_date": str(max_precip_date),
    "min_temp_c": min_temp,
    "temp_precip_correlation": correlation,
    "average_precip_mm": avg_precip,
    "temp_line_chart": temp_base64,
    "precip_histogram": precip_base64
}

# Write the JSON object to result.json
with open('result.json', 'w') as f:
    json.dump(result, f)