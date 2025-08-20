# Import necessary libraries
import pandas as pd

# Load the dataframe
df = pd.read_csv('sample-weather.csv')

# Open metadata.txt in append mode
with open('metadata.txt', 'a') as f:
    # Write column names
    f.write(str(df.columns) + '\n')
    # Write the first 3 rows
    f.write(str(df.head(3)) + '\n')