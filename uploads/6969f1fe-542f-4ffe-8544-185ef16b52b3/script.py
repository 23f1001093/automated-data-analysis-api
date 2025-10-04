import pandas as pd
import json
from matplotlib import pyplot as plt
import base64
import io

# Load the dataframe.
df = pd.read_csv('sample-sales.csv')

# Initial data exploration. Column names and first 5 rows are printed to metadata.txt
with open('metadata.txt', 'w') as f:
    f.write('Columns:\n')
    f.write(str(df.columns) + '\n')
    f.write('\nFirst 5 rows:\n')
    f.write(str(df.head()) + '\n')

# 1. Total sales
# Verified column name from metadata.txt: 'Total'
total_sales = df['Total'].sum()

# 2. Region with highest sales
sales_by_region = df.groupby('Region')['Total'].sum()
top_region = sales_by_region.idxmax()

# 3. Correlation between day and sales
# Verified column name from metadata.txt: 'Order Date'
df['Date'] = pd.to_datetime(df['Order Date'])
df['Day'] = df['Date'].dt.day
day_sales_correlation = df['Day'].corr(df['Total'])

# 4. Bar chart of sales by region
plt.figure(figsize=(8, 6))
plt.bar(sales_by_region.index, sales_by_region.values, color='blue')
plt.xlabel('Region')
plt.ylabel('Sales')
plt.title('Total Sales by Region')
buffer = io.BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
base64_bar = base64.b64encode(buffer.read()).decode('utf-8')
plt.clf()

# 5. Median sales
median_sales = df['Total'].median()

# 6. Total sales tax
tax_rate = 0.1
total_sales_tax = total_sales * tax_rate

# 7. Cumulative sales chart
df['Cumulative Sales'] = df['Total'].cumsum()
plt.figure(figsize=(8, 6))
plt.plot(df['Date'], df['Cumulative Sales'], color='red')
plt.xlabel('Date')
plt.ylabel('Cumulative Sales')
plt.title('Cumulative Sales Over Time')
buffer = io.BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
base64_cumulative = base64.b64encode(buffer.read()).decode('utf-8')
plt.clf()

# Create the dictionary
result_dict = {
    'total_sales': float(total_sales),
    'top_region': top_region,
    'day_sales_correlation': float(day_sales_correlation),
    'bar_chart': base64_bar,
    'median_sales': float(median_sales),
    'total_sales_tax': float(total_sales_tax),
    'cumulative_sales_chart': base64_cumulative
}

# Save to result.json
with open('result.json', 'w') as f:
    json.dump(result_dict, f)
