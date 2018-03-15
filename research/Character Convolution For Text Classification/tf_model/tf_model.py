import multiprocessing

import tensorflow as tf
from tensorflow.contrib import lookup
HEADERS = ['label', 'text']
FEATURE_COL = 'text'
LABEL_COL = 'label'
TARGET_SIZE = 4
HEADER_DEFAULTS = [['NA'], ['NA']]


__author__ = 'KKishore'

tf.logging.set_verbosity(tf.logging.INFO)


def parse_csv_row(row):
    columns = tf.decode_csv(row, record_defaults=HEADER_DEFAULTS)
    features = dict(zip(HEADERS, columns))
    target = features.pop(LABEL_COL)
    target = tf.string_to_number(target, out_type=tf.int32)
    return features, target


def input_fn(file_name, batch_size=128, shuffle=False, repeat_count=1):
    num_threads = multiprocessing.cpu_count()

    data_set = tf.data.TextLineDataset(filenames=file_name)
    data_set = data_set.skip(1)

    if shuffle:
        data_set = data_set.shuffle(buffer_size=1000)

    data_set = data_set.map(lambda row: parse_csv_row(row), num_parallel_calls=num_threads).batch(batch_size) \
        .repeat(repeat_count).prefetch(1000)

    iterator = data_set.make_one_shot_iterator()
    features, target = iterator.get_next()
    return features, target


def model_fn(features, labels, mode, params):
    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.keras.backend.set_learning_phase(True)
    else:
        tf.keras.backend.set_learning_phase(False)
    text = features[FEATURE_COL]
    words = tf.string_split(text)
    x = words.values
    split_chars = tf.string_split(x, delimiter='')
    rev = tf.reverse(split_chars.values, axis=[-1])
    table = lookup.index_table_from_file(vocabulary_file='v.txt', vocab_size=69, default_value=0)

    word_ids = table.lookup(rev)
    word_ids = tf.expand_dims(word_ids, axis=[0])

    padding = tf.constant([[0, 0], [0, 1014]])
    # Pad all the word_ids entries to the maximum document length
    word_ids_padded = tf.pad(word_ids, padding)
    word_id_vector = tf.slice(word_ids_padded, [0, 0], [-1, 1014])

    #input = tf.cast(word_id_vector, dtype=tf.float32)
    #input = tf.expand_dims(input, axis=[0])

    f1 = tf.keras.layers.Embedding(69, 128, input_length=1014)(word_id_vector)
    f1 = tf.keras.layers.Convolution1D(filters=128, kernel_size=7, padding="valid", activation='relu')(f1)
    f1 = tf.keras.layers.MaxPooling1D(pool_size=3)(f1)
    f1 = tf.keras.layers.Flatten()(f1)
    #f1 = tf.keras.layers.Flatten()(f1)
    logits = tf.keras.layers.Dense(4, activation=None)(f1)

    predictions = tf.nn.softmax(logits)
    prediction_indices = tf.argmax(logits, axis=1)

    onehot_labels = tf.one_hot(labels, 4, 1, 0)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    #loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    print(tf.shape(labels))
    print(tf.shape(prediction_indices))
    eval_metrics_ops = {
        'accuracy': tf.metrics.accuracy(labels=labels, predictions=prediction_indices)
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metrics_ops)


estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir='build/')

estimator.train(input_fn=lambda: input_fn('data/ag_news_csv/train.csv', shuffle=True, repeat_count=5))
