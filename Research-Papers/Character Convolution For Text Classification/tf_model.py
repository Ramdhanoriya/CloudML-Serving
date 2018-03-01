import multiprocessing

import tensorflow as tf
from tensorflow.contrib import lookup

from model import commons

__author__ = 'KKishore'

table = lookup.index_table_from_file(vocabulary_file='data/vocab.csv', vocab_size=69, default_value=0)


def parse_csv_row(row):
    columns = tf.decode_csv(row, record_defaults=commons.HEADER_DEFAULTS)
    features = dict(zip(commons.HEADERS, columns))
    target = features.pop(commons.LABEL_COL)
    target = tf.cast(target, dtype=tf.int32)
    words = tf.string_split([features[commons.FEATURE_COL]])
    x = words.values
    split_chars = tf.string_split(x, delimiter='')
    encoded = tf.one_hot(table.lookup(split_chars.values), commons.MAX_DOCUMENT_LENGTH, dtype=tf.float32)
    features = {commons.FEATURE_COL: encoded}
    return features, target


def input_fn(file_name, batch_size=16, shuffle=False, repeat_count=1):
    num_threads = multiprocessing.cpu_count()

    data_set = tf.data.TextLineDataset(filenames=file_name)
    data_set = data_set.skip(1)

    if shuffle:
        data_set = data_set.shuffle(buffer_size=1000)

    data_set = data_set.map(lambda row: parse_csv_row(row), num_parallel_calls=num_threads).batch(batch_size) \
        .repeat(repeat_count).prefetch(1000)

    iterator = data_set.make_initializable_iterator()
    # iterator = data_set.make_one_shot_iterator()
    features, target = iterator.get_next()
    return features, target


def model_fn(features, labels, mode, params):
    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.keras.backend.set_learning_phase(True)
    else:
        tf.keras.backend.set_learning_phase(False)
    text = features[commons.FEATURE_COL]
    '''
    words = tf.string_split(text)
    x = words.values
    split_chars = tf.string_split(x, delimiter='')
    table = lookup.index_table_from_file(vocabulary_file='vocab.csv', vocab_size=69, default_value=0)
    encoded = tf.one_hot(table.lookup(split_chars.values), commons.MAX_DOCUMENT_LENGTH, dtype=tf.float32)
    encoded = tf.expand_dims(encoded, axis=1)
    '''
    f1 = tf.keras.layers.Convolution1D(filters=256, kernel_size=7, padding="valid", activation='relu')(text)
    f1 = tf.keras.layers.MaxPooling1D(pool_size=3)(f1)
    f1 = tf.keras.layers.Flatten()(f1)
    logits = tf.keras.layers.Dense(commons.TARGET_SIZE, activation=None)(f1)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    eval_metrics_ops = {
        'accuracy': tf.metrics.accuracy(labels=labels, predictions=logits)
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metrics_ops)


estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir='build/')

estimator.train(input_fn=lambda: input_fn('data/train-data.csv', shuffle=True, repeat_count=5))
