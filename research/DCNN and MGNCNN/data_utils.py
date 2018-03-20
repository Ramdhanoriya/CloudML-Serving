import re

import pandas as pd
from bs4 import BeautifulSoup


data_set = pd.read_csv('D:\\DataSet\\Kaggle Sentiment Analysis\\labeledTrainData.tsv', sep='\t')


def clean(text):
    s = str(text)

    s = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", s)

    s = re.sub(r"\'s", " \'s", s)

    s = re.sub(r"\'ve", " \'ve", s)

    s = re.sub(r"n\'t", " n\'t", s)

    s = re.sub(r"\'re", " \'re", s)

    s = re.sub(r"\'d", " \'d", s)

    s = re.sub(r"\'ll", " \'ll", s)

    s = re.sub(r",", " , ", s)

    s = re.sub(r"!", " ! ", s)

    s = re.sub(r"\(", " \( ", s)

    s = re.sub(r"\)", " \) ", s)

    s = re.sub(r"\?", " \? ", s)

    s = re.sub(r"\\", "", s)

    s = re.sub(r"\)", "", s)

    s = re.sub(r"\(", "", s)

    s = re.sub(r"\}", "", s)

    s = re.sub(r"\{", "", s)

    s = re.sub(r"\?", "", s)

    s = re.sub(r"\s{2,}", " ", s)

    s = re.sub(r'[^\x00-\x7F]+', "", s)

    s = re.sub(r'"', "", s)

    s = re.sub(r' , ', ' ', s)
    s = re.sub(r' \'', '\'', s)
    s = re.sub(r'`', '', s)
    s = re.sub(r'br', '', s)
    s = re.sub(r'   ', ' ', s)

    s = s.strip().lower()
    s = BeautifulSoup(s, "lxml").text
    return s


data_set['review'] = data_set['review'].apply(lambda x: clean(x))
data_set.drop('id', axis=1, inplace=True)

data_set.to_csv('data/train.tsv', sep='\t', encoding='utf-8', index=False)
