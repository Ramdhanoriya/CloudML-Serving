from keras.layers import Convolution1D
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Input, Dense, Flatten
from keras.layers import MaxPooling1D
from keras.layers import ThresholdedReLU
from keras.models import Model
from keras.optimizers import Adam

print("Loading the configurations...", )

conv_layers = conv_layers = [[256, 7, 3],
                             [256, 7, 3],
                             [256, 3, None],
                             [256, 3, None],
                             [256, 3, None],
                             [256, 3, 3]]
fully_layers = [1024, 1024]
l0 = 1014
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
alphabet_size = len(alphabet)

embedding_size = 128
num_of_classes = 4
th = 1e-6
dropout_p = 0.5

base_rate = 1e-2
momentum = 0.9
decay_step = 15000
decay_rate = 0.95
epochs = 10
evaluate_every = 100
checkpoint_every = 100
batch_size = 128

print("Loaded")

print("Building the model..."),
# building the model

# Input layer
inputs = Input(shape=(l0,), name='sent_input', dtype='int64')

# Embedding layer

x = Embedding(alphabet_size + 1, embedding_size, input_length=l0)(inputs)

# Convolution layers
for cl in conv_layers:
    x = Convolution1D(cl[0], cl[1])(x)
    x = ThresholdedReLU(th)(x)
    if not cl[2] is None:
        x = MaxPooling1D(cl[2])(x)

x = Flatten()(x)

# Fully connected layers

for fl in fully_layers:
    x = Dense(fl)(x)
    x = ThresholdedReLU(th)(x)
    x = Dropout(0.5)(x)

predictions = Dense(num_of_classes, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)

optimizer = Adam()

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

print("Built")

print("Loading the data sets...", )

from keras_model.data_utils import Data

train_data = Data(data_source='data/ag_news_csv/train.csv', alphabet=alphabet, l0=l0, batch_size=0,
                  no_of_classes=num_of_classes)

train_data.loadData()

X_train, y_train = train_data.getAllData()

dev_data = Data(data_source='data/ag_news_csv/test.csv', alphabet=alphabet, l0=l0, batch_size=0,
                no_of_classes=num_of_classes)

dev_data.loadData()

X_val, y_val = dev_data.getAllData()

print("Loadded")

print("Training ...")

model.fit(X_train, y_train, nb_epoch=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=1)

print("Done!.")
