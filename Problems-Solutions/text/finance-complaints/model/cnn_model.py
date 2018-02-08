import os

import tensorflow as tf


def model_estimator(vocab_size, max_length, model_dir=os.getcwd() + '\\' + 'build'):
    '''
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
    '''

    sequence_input = tf.keras.layers.Input(shape=(max_length,), dtype='int32')
    embedded_sequences = tf.keras.layers.Embedding(vocab_size, 128, input_length=max_length)(sequence_input)

    '''
    convs = []
    filter_sizes = [3, 4, 5]

    print('Filters..... \n')

    for filter_size in filter_sizes:
        l_conv = tf.keras.layers.Conv1D(filters=128, kernel_size=filter_size, activation='relu')(embedded_sequences)
        l_pool = tf.keras.layers.MaxPooling1D(pool_size=3)(l_conv)
        convs.append(l_pool)

    l_merge = tf.keras.layers.concatenate(convs, axis=1)
    '''
    conv = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu')(embedded_sequences)
    pool = tf.keras.layers.MaxPooling1D(pool_size=3)(conv)
    f1 = tf.keras.layers.Dropout(0.5)(pool)
    f1 = tf.keras.layers.Flatten()(f1)
    f1 = tf.keras.layers.Dense(128, activation='relu')(f1)
    logits = tf.keras.layers.Dense(11, activation='softmax')(f1)

    model = tf.keras.models.Model(inputs=sequence_input, outputs=logits)
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0002),
                  metrics=['accuracy', tf.keras.metrics.categorical_accuracy])
    return tf.keras.estimator.model_to_estimator(model, model_dir=model_dir)
