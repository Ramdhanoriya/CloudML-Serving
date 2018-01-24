import os

import tensorflow as tf


def model_estimator(vocab_size, max_length, model_dir=os.getcwd() + '\\' + 'build'):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, 50, input_length=max_length, name='text'))
    model.add(tf.keras.layers.Conv1D(filters=50, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.GlobalMaxPooling1D())
    #model.add(tf.keras.layers.Average(submodels))
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(11, activation='softmax', name='output'))
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy', tf.keras.metrics.categorical_accuracy])
    return tf.keras.estimator.model_to_estimator(model, model_dir=model_dir)
