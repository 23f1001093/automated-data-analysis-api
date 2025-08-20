import pandas as pd

df = pd.read_csv('sample-sales.csv')

with open('metadata.txt', 'a') as f:
    f.write(str(df.columns) + '\n')
    f.write(str(df.head(3)) + '\n')