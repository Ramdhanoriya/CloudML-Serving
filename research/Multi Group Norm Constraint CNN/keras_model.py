from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils

import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split

# set parameters:
maxlen = 50
batch_size = 16
embedding_dims = 30
filters = 250
kernel_size = 3
hidden_dims = 256
epochs = 10


data_set = pd.read_csv('data/train.tsv', sep='\t')
x_train = data_set['review'].values
y_train = data_set['sentiment'].values

tokenizer = Tokenizer(num_words=20000)

tokenizer.fit_on_texts(x_train)

x_train = tokenizer.texts_to_sequences(x_train)


print(len(x_train), 'train sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
print('x_train shape:', x_train.shape)

y_train = y_train.astype(np.int32)
y_train = np_utils.to_categorical(y_train, num_classes=2)

print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(20000, embedding_dims, input_length=maxlen))
model.add(Dropout(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.25)