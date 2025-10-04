import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Load the dataframe
df = pd.read_csv('sample-sales.csv')

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])

# Calculate total sales
total_sales = df['sales'].sum()

# Calculate sales by region
sales_by_region = df.groupby('region')['sales'].sum()

# Find top region
top_region = sales_by_region.idxmax()

# Calculate correlation between day of month and sales
df['day_of_month'] = df['date'].dt.day
day_sales_correlation = df['day_of_month'].corr(df['sales'])

# Create bar chart of sales by region
plt.figure(figsize=(8, 6))
plt.bar(sales_by_region.index, sales_by_region.values, color='blue')
plt.xlabel('Region')
plt.ylabel('Sales')
plt.title('Total Sales by Region')
buffer = BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
base64_bar = base64.b64encode(buffer.read()).decode('utf-8')
plt.close()

# Calculate median sales
median_sales = df['sales'].median()

# Calculate total sales tax
tax_rate = 0.10
total_sales_tax = total_sales * tax_rate

# Create line chart of cumulative sales
plt.figure(figsize=(8, 6))
plt.plot(df['date'], df['sales'].cumsum(), color='red')
plt.xlabel('Date')
plt.ylabel('Cumulative Sales')
plt.title('Cumulative Sales Over Time')
buffer = BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
base64_cumulative = base64.b64encode(buffer.read()).decode('utf-8')
plt.close()

# Store results in dictionary then save to json file
results = {
    "total_sales": int(total_sales),
    "top_region": top_region,
    "day_sales_correlation": float(day_sales_correlation),
    "bar_chart": base64_bar,
    "median_sales": int(median_sales),
    "total_sales_tax": float(total_sales_tax),
    "cumulative_sales_chart": base64_cumulative
}

with open('result.json', 'w') as f:
    import json
    json.dump(results, f)