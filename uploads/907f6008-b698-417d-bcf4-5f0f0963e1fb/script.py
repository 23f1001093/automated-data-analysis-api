import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import os

# Load the dataframe.
df = pd.read_csv('sample-sales.csv')

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])

# 1. Total sales
total_sales = df['sales'].sum()

# 2. Top region
top_region = df.groupby('region')['sales'].sum().idxmax()

# 3. Correlation between day of month and sales
df['day_of_month'] = df['date'].dt.day
day_sales_correlation = df['day_of_month'].corr(df['sales'])

# 4. Bar chart of total sales by region
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(df.groupby('region')['sales'].sum().index, df.groupby('region')['sales'].sum().values, color='blue')
ax.set_xlabel('Region')
ax.set_ylabel('Total Sales')
ax.set_title('Total Sales by Region')
buf = io.BytesIO()
fig.savefig(buf, format='png')
buf.seek(0)
bar_chart = base64.b64encode(buf.read()).decode('utf-8')
buf.close()

# 5. Median sales
median_sales = df['sales'].median()

# 6. Total sales tax
tax_rate = 0.10
total_sales_tax = total_sales * tax_rate

# 7. Cumulative sales chart
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(df['date'], df['sales'].cumsum(), color='red')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Sales')
ax.set_title('Cumulative Sales Over Time')
buf = io.BytesIO()
fig.savefig(buf, format='png')
buf.seek(0)
cumulative_sales_chart = base64.b64encode(buf.read()).decode('utf-8')
buf.close()

# Save to result.json
result = {
    'total_sales': int(total_sales),
    'top_region': top_region,
    'day_sales_correlation': float(day_sales_correlation),
    'bar_chart': bar_chart,
    'median_sales': int(median_sales),
    'total_sales_tax': float(total_sales_tax),
    'cumulative_sales_chart': cumulative_sales_chart
}

import json
with open('result.json', 'w') as f:
    json.dump(result, f)