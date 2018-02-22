import pandas as pd
import numpy as np

train_df = pd.read_csv("data/train.csv", sep='\t').fillna("sterby")

msk = np.random.rand(len(train_df)) < 0.9

train = train_df[msk]
test = train_df[~msk]

train.to_csv('data/train_m.csv', index=False, encoding='utf-8')
test.to_csv('data/test_m.csv', index=False, encoding='utf-8')
