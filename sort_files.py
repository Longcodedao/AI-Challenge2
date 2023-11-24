import pandas as pd

df = pd.read_csv('./data/META.csv')

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by = 'Date')

print(df.head(5))

df.to_csv('./data/GOOG.csv', index = False)