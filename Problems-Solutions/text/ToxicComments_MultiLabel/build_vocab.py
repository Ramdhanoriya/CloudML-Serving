__author__ = 'KKishore'

import pandas as pd
import re

from model.commons import FEATURE_COL, PAD_WORD

def clean_str(x):
    s = x
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
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r'\S*(x{2,}|X{2,})\S*', "xxx", s)
    s = re.sub(r'[^\x00-\x7F]+', "", s)
    s = re.sub(r'"', "", s)
    s = s.strip().lower()
    return s

def build_vocab(file_name):
    data_set = pd.read_csv(file_name, sep='\t')    
    sentences = data_set[FEATURE_COL].values
    vocab_set = set()
    for sentence in sentences:
        text = str(sentence)        
        words = text.split(' ')            
        word_set = set(words)
        vocab_set.update(word_set)        
    return list(vocab_set)


vocab_list = build_vocab('data/train_preprocess.csv')

with open('data/vocab.csv', 'w', encoding='utf-8') as vocab_file:
    vocab_file.write("{}\n".format(PAD_WORD))
    for word in vocab_list:
        vocab_file.write("{}\n".format(word))

with open('data/nwords.csv', mode='w') as n_words:
    n_words.write(str(len(vocab_list)))
