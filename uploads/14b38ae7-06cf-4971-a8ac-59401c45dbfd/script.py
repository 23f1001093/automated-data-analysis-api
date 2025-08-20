import pandas as pd
import matplotlib.pyplot as plt
import base64
import json
from io import BytesIO

# Load the dataframe.
df = pd.read_csv('sample-sales.csv')

# Preliminary data exploration (Step 1)
metadata = open('metadata.txt', 'w')
metadata.write(str(df.head()))
metadata.close()

# Calculate total sales
total_sales = df['Sales'].sum()

# Find top region
top_region = df.groupby('Region')['Sales'].sum().idxmax()

# Calculate correlation
df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df['Date'].dt.day
day_sales_correlation = df['Day'].corr(df['Sales'])

# Create bar chart
plt.figure(figsize=(8, 6))
plt.bar(df['Region'].unique(), df.groupby('Region')['Sales'].sum(), color='blue')
plt.xlabel('Region')
plt.ylabel('Total Sales')
plt.title('Total Sales by Region')
buffer = BytesIO()
plt.savefig(buffer, format='png')
plt.close()
bar_chart = base64.b64encode(buffer.getvalue()).decode('utf-8')

# Calculate median sales
median_sales = df['Sales'].median()

# Calculate total sales tax
total_sales_tax = total_sales * 0.1

# Create cumulative sales chart
df['Cumulative_Sales'] = df['Sales'].cumsum()
plt.figure(figsize=(8, 6))
plt.plot(df['Date'], df['Cumulative_Sales'], color='red')
plt.xlabel('Date')
plt.ylabel('Cumulative Sales')
plt.title('Cumulative Sales Over Time')
buffer = BytesIO()
plt.savefig(buffer, format='png')
plt.close()
cumulative_sales_chart = base64.b64encode(buffer.getvalue()).decode('utf-8')

# Save results to JSON
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
