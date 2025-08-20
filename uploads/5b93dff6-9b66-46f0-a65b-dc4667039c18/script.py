import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from io import BytesIO
import base64

# Load the dataframe.
df = pd.read_csv('sample-sales.csv')

# Convert 'Order Date' or 'OrderDate' to datetime
date_col = 'Order Date' if 'Order Date' in df.columns else 'OrderDate'
df[date_col] = pd.to_datetime(df[date_col])

# Extract day of month
df['Day'] = df[date_col].dt.day

# 1. Total sales
total_sales = df['Sales'].sum()

# 2. Region with highest sales
region_sales = df.groupby('Region')['Sales'].sum()
top_region = region_sales.idxmax()

# 3. Correlation between day and sales
day_sales_correlation = df['Day'].corr(df['Sales'])

# 4. Bar chart of sales by region
plt.figure(figsize=(8, 6))
sns.barplot(x='Region', y='Sales', data=df, color='blue', estimator=sum)
plt.title('Total Sales by Region')
buffer = BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
bar_chart = base64.b64encode(buffer.read()).decode('utf-8')
plt.close()

# 5. Median sales
median_sales = df['Sales'].median()

# 6. Total sales tax
tax_rate = 0.1
total_sales_tax = total_sales * tax_rate

# 7. Cumulative sales chart
df['Cumulative Sales'] = df['Sales'].cumsum()
plt.figure(figsize=(10, 6))
sns.lineplot(x=date_col, y='Cumulative Sales', data=df, color='red')
plt.title('Cumulative Sales Over Time')
buffer = BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
cumulative_sales_chart = base64.b64encode(buffer.read()).decode('utf-8')
plt.close()

# Create the JSON object
result = {
    "total_sales": total_sales,
    "top_region": top_region,
    "day_sales_correlation": day_sales_correlation,
    "bar_chart": bar_chart,
    "median_sales": median_sales,
    "total_sales_tax": total_sales_tax,
    "cumulative_sales_chart": cumulative_sales_chart
}

# Write the JSON object to result.json
with open('result.json', 'w') as f:
    json.dump(result, f)
