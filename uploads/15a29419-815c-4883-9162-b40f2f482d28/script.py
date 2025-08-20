import pandas as pd
import json
from matplotlib import pyplot as plt
import base64
import io

# Load the dataframe.
df = pd.read_csv('sample-sales.csv')

# Initial data exploration (already done in previous step, no need to repeat)

# 1. Total sales
total_sales = df['sales'].sum()

# 2. Top region
top_region = df.groupby('region')['sales'].sum().idxmax()

# 3. Correlation
df['date'] = pd.to_datetime(df['date'])
df['day_of_month'] = df['date'].dt.day
day_sales_correlation = df['day_of_month'].corr(df['sales'])

# 4. Bar chart
plt.figure(figsize=(8, 6))
region_sales = df.groupby('region')['sales'].sum()
plt.bar(region_sales.index, region_sales.values, color='blue')
plt.xlabel('Region')
plt.ylabel('Total Sales')
plt.title('Total Sales by Region')
buffer = io.BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
bar_chart = base64.b64encode(buffer.read()).decode('utf-8')
plt.close()

# 5. Median sales
median_sales = df['sales'].median()

# 6. Total sales tax
tax_rate = 0.10
total_sales_tax = total_sales * tax_rate

# 7. Cumulative sales chart
df = df.sort_values('date')
df['cumulative_sales'] = df['sales'].cumsum()
plt.figure(figsize=(8, 6))
plt.plot(df['date'], df['cumulative_sales'], color='red')
plt.xlabel('Date')
plt.ylabel('Cumulative Sales')
plt.title('Cumulative Sales Over Time')
buffer = io.BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
cumulative_sales_chart = base64.b64encode(buffer.read()).decode('utf-8')
plt.close()

# Create and save the result dictionary to a JSON file
result = {
    'total_sales': total_sales,
    'top_region': top_region,
    'day_sales_correlation': day_sales_correlation,
    'bar_chart': bar_chart,
    'median_sales': median_sales,
    'total_sales_tax': total_sales_tax,
    'cumulative_sales_chart': cumulative_sales_chart
}
with open('result.json', 'w') as f:
    json.dump(result, f)
