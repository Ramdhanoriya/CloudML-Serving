import pickle

import numpy as np
import pandas as pd
from keras import layers
from keras.models import Model, Input
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import json
import csv

train_df = pd.read_csv("data/train_m.csv").fillna("sterby")

X_train = train_df["comment_text"].values
y_train = train_df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

max_features = 2000 #5000  # number of words we want to keep
maxlen = 100 #200  # max length of the comments in the model
batch_size = 64 #32  # batch size for the model
embedding_dims = 20 #50  # dimension of the hidden variable, i.e. the embedding dimension

filters = 64
kernel_size = 3
hidden_dims = 32

tok = Tokenizer(num_words=max_features)
tok.fit_on_texts(list(X_train))

with open('word_index.json', 'w') as ff:
    json.dump(tok.word_index, ff)

key_list = tok.word_index.keys()
print(key_list)
with open("output.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(key_list)

x_train = tok.texts_to_sequences(X_train)
print(len(x_train), 'train sequences')
print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print(y_train[0])


def fast_text():
    comment_input = Input((maxlen,))
    comment_emb = layers.Embedding(max_features, embedding_dims, input_length=maxlen, embeddings_initializer="uniform")(
        comment_input)
    h = layers.GlobalMaxPooling1D()(comment_emb)
    output = layers.Dense(6, activation='sigmoid')(h)
    model = Model(inputs=comment_input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002), metrics=['accuracy'])
    return model


def cnn_model():
    sequence_input = layers.Input(shape=(maxlen,), dtype='int32')
    embedded_sequences = layers.Embedding(max_features, embedding_dims, input_length=maxlen)(sequence_input)
    conv = layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(embedded_sequences)
    pool = layers.GlobalMaxPool1D()(conv)
    f1 = layers.Dense(hidden_dims)(pool)
    f1 = layers.Dropout(0.5)(f1)
    f1 = layers.Activation(activation='relu')(f1)
    # f1 = layers.Flatten()(f1)
    logits = layers.Dense(6, activation='sigmoid')(f1)

    model = Model(inputs=sequence_input, outputs=logits)

    print(model.summary())

    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002), metrics=['acc'])
    return model


model = fast_text()
model.fit(x_train, y_train, batch_size=batch_size, epochs=8, validation_split=0.10)

with open('tokenizer.pkl', 'wb') as token_serialize:
    pickle.dump(obj=tok, file=token_serialize, protocol=pickle.HIGHEST_PROTOCOL)

model.save('fasttext.h5')

print('Model Building is Done !!')
