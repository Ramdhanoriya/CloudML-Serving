import pandas as pd
import numpy as np

data_set = pd.read_csv('data/org-train.tsv', sep='\t')

msk = np.random.rand(len(data_set)) < 0.8

train_m = data_set[msk]

dev = data_set[~msk]

train_m.to_csv('data/train_1.tsv', sep='\t', encoding='utf-8', index=False)
dev.to_csv('data/test.tsv', sep='\t', encoding='utf-8', index=False)


