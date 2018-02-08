import numpy as np
import pandas as pd
import tensorflow as tf

selected = ['product', 'consumer_complaint_narrative']


def input_function(features, labels, shuffle=False, batch_size=128, epochs=1):
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'input_1': np.array(features, dtype=np.float32)},
        y=np.array(labels, dtype=np.float32),
        shuffle=shuffle,
        batch_size=batch_size,
        num_epochs=epochs
    )

    return input_fn


def get_data_set(file_name):
    from pandas.api.types import is_numeric_dtype
    data_set = pd.read_csv(file_name, header=None, sep=',', names=selected)
    data_set['consumer_complaint_narrative'] = data_set['consumer_complaint_narrative'].astype(str)
    data_set['product'] = pd.to_numeric(data_set['product'], downcast='float')
    print(is_numeric_dtype(data_set['product']))
    return data_set['consumer_complaint_narrative'].values, data_set['product'].values


def enocde_text(tokenizer, data_set, max_length):
    encoded = tokenizer.texts_to_sequences(data_set)
    padded = tf.keras.preprocessing.sequence.pad_sequences(encoded, maxlen=max_length, padding='post')
    return padded
