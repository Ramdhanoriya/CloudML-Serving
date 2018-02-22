import pandas as pd

df=pd.read_json("word_index.json")

df.to_csv('vocab.csv', index=False)
