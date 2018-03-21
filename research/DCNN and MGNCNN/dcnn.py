from keras.datasets import imdb
from keras.layers import Conv1D, Flatten, ZeroPadding1D
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.models import Sequential
from keras.preprocessing import sequence
from utils import KMaxPooling, Folding

# set parameters:
max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
filters = 64
kernel_size = 50
epochs = 5

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
model.add(Dropout(0.2))
model.add(ZeroPadding1D((49, 49)))
model.add(Conv1D(filters, kernel_size, padding='same', activation=None, strides=1))
model.add(KMaxPooling(k=9, axis=1))

model.add(ZeroPadding1D((24,24)))
model.add(Conv1D(filters, 25, padding='same', activation=None, strides=1))
model.add(Folding())
model.add(KMaxPooling(k=9, axis=1))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dropout(0.2))

model.add(Dense(1))
model.add(Activation('sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
