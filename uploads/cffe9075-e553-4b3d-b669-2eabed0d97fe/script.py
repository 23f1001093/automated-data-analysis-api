import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

df = pd.read_csv('sample-weather.csv')

with open('metadata.txt', 'a') as f:
    f.write(str(df.columns) + '\n')
    f.write(str(df.head(3)) + '\n')