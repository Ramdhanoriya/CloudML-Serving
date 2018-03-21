from keras.preprocessing.text import Tokenizer
import pandas as pd

PAD_WORD = 'UNK'

data_set = pd.read_csv('data/train.tsv', sep='\t')
data = data_set['review'].values

num_words = 5000

token = Tokenizer(oov_token='UNK', num_words=num_words+1)
token.fit_on_texts(data)
token.word_index = {e:i for e,i in token.word_index.items() if i <= num_words}
token.word_index[token.oov_token] = num_words + 1

word_dict = token.word_index

print(len(word_dict))

items = word_dict.keys()

with open('data/vocab.csv', 'w', encoding='utf-8') as vocab_file:
    vocab_file.write("{}\n".format(PAD_WORD))
    for word in items:
        vocab_file.write("{}\n".format(word))

with open('data/nwords.csv', mode='w') as n_words:
    n_words.write(str(len(items)))