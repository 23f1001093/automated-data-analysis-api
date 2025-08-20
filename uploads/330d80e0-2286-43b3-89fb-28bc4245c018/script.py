import pandas as pd
import matplotlib.pyplot as plt
import json
from io import BytesIO
import base64
import os

# Get the absolute path of the current working directory
current_dir = os.getcwd()

# Construct the full path to the CSV file
csv_file_path = os.path.join(current_dir, 'sample-sales.csv')

# Read the CSV file into a pandas DataFrame
try:
    df = pd.read_csv(csv_file_path)
except FileNotFoundError:
    print(f"Error: File not found at {csv_file_path}")
    exit()

# Initial data exploration (Step 1)
with open('metadata.txt', 'w') as metadata_file:
    metadata_file.write(str(df.head()))

# Calculate total sales
total_sales = df['Sales'].sum()

# Find the region with the highest sales
top_region = df.groupby('Region')['Sales'].sum().idxmax()

# Calculate correlation between day of month and sales
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
buffer.seek(0)
bar_chart = base64.b64encode(buffer.read()).decode('utf-8')
plt.close()

# Calculate median sales
median_sales = df['Sales'].median()

# Calculate total sales tax
total_sales_tax = total_sales * 0.1

# Create cumulative sales chart
df['Cumulative Sales'] = df['Sales'].cumsum()
plt.figure(figsize=(8, 6))
plt.plot(df['Date'], df['Cumulative Sales'], color='red')
plt.xlabel('Date')
plt.ylabel('Cumulative Sales')
plt.title('Cumulative Sales Over Time')
buffer = BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
cumulative_sales_chart = base64.b64encode(buffer.read()).decode('utf-8')
plt.close()

# Save results to result.json
result = {
    "total_sales": total_sales,
    "top_region": top_region,
    "day_sales_correlation": day_sales_correlation,
    "bar_chart": bar_chart,
    "median_sales": median_sales,
    "total_sales_tax": total_sales_tax,
    "cumulative_sales_chart": cumulative_sales_chart
}
with open('result.json', 'w') as f:
    json.dump(result, f)
