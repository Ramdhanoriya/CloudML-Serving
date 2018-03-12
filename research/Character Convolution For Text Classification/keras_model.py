from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.models import Model
from keras.optimizers import SGD


def model(filter_kernels, dense_outputs, maxlen, vocab_size, nb_filter,
          cat_output):
    # Define what the input shape looks like
    inputs = Input(shape=(maxlen, vocab_size), name='input', dtype='float32')

    # All the convolutional layers...
    conv = Convolution1D(filters=nb_filter, kernel_size=filter_kernels[0], padding="valid", activation='relu',
                         input_shape=(maxlen, vocab_size))(inputs)
    conv = MaxPooling1D(pool_size=3)(conv)

    conv1 = Convolution1D(filters=nb_filter, kernel_size=filter_kernels[1], padding='valid', activation='relu')(conv)
    conv1 = MaxPooling1D(pool_size=3)(conv1)

    conv2 = Convolution1D(filters=nb_filter, kernel_size=filter_kernels[2], padding='valid', activation='relu')(conv1)

    conv3 = Convolution1D(filters=nb_filter, kernel_size=filter_kernels[3], padding='valid', activation='relu')(conv2)

    conv4 = Convolution1D(filters=nb_filter, kernel_size=filter_kernels[4], padding='valid', activation='relu')(conv3)

    conv5 = Convolution1D(filters=nb_filter, kernel_size=filter_kernels[5], padding='valid', activation='relu')(conv4)
    conv5 = MaxPooling1D(pool_size=3)(conv5)
    conv5 = Flatten()(conv5)

    # Two dense layers with dropout of .5
    z = Dropout(0.5)(Dense(dense_outputs, activation='relu')(conv5))
    z = Dropout(0.5)(Dense(dense_outputs, activation='relu')(z))

    # Output dense layer with softmax activation
    pred = Dense(cat_output, activation='softmax', name='output')(z)

    model = Model(inputs=inputs, outputs=pred)

    sgd = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model
