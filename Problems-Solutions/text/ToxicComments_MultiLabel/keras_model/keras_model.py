import pandas as pd
import numpy as np

from keras.preprocessing import sequence
from keras.models import Model, Input
from keras.layers import Dense, Embedding, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import pickle

train_df = pd.read_csv("data/train.csv").fillna("sterby")
#test_df = pd.read_csv("data/test.csv").fillna("sterby")

X_train = train_df["comment_text"].values
y_train = train_df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

train_x, test_x, train_y, test_y = train_test_split(X_train, y_train, test_size=0.20)

#X_test = test_df["comment_text"].values

max_features = 30000  # number of words we want to keep
maxlen = 100  # max length of the comments in the model
batch_size = 64  # batch size for the model
embedding_dims = 75  # dimension of the hidden variable, i.e. the embedding dimension

tok = Tokenizer(num_words=max_features)
tok.fit_on_texts(list(train_x))
x_train = tok.texts_to_sequences(train_x)
print(len(x_train), 'train sequences')
print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print(train_y[0])

comment_input = Input((maxlen,))

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
comment_emb = Embedding(max_features, embedding_dims, input_length=maxlen, embeddings_initializer="uniform")(comment_input)

# we add a GlobalMaxPooling1D, which will extract features from the embeddings
# of all words in the comment
h = GlobalMaxPooling1D()(comment_emb)

# We project onto a six-unit output layer, and squash it with a sigmoid:
output = Dense(6, activation='sigmoid')(h)

model = Model(inputs=comment_input, outputs=output)

model.compile(loss='binary_crossentropy', optimizer=Adam(0.01), metrics=['accuracy'])

hist = model.fit(x_train, train_y, batch_size=batch_size, epochs=1, validation_split=0.10)

with open('tokenizer.pkl', 'wb') as token_serialize:
    pickle.dump(obj=tok, file=token_serialize, protocol=pickle.HIGHEST_PROTOCOL)

model.save('fasttext.h5')

print('Model Building is Done !!')

threshold = [0.2, 0.25, 0.3, 0.35, 0.4]

print('Trying Various Epoch !!')

for point in threshold:
    print('For Point = ', point)
    x_test = tok.texts_to_sequences(test_x)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print(model.predict(x=x_test, batch_size=1)[0])
    #f1_score()



