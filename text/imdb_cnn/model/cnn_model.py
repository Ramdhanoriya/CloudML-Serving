import multiprocessing

import tensorflow as tf
from tensorflow.contrib import lookup, layers

from model import commons

__author__ = 'KKishore'


def parse_csv_row(row):
    columns = tf.decode_csv(row, record_defaults=commons.HEADER_DEFAULTS, field_delim='\t')
    features = dict(zip(commons.HEADERS, columns))
    target = features.pop(commons.LABEL_COL)
    return features, tf.string_to_number(target, out_type=tf.int32)


def input_fn(file_name, batch_size=32, shuffle=False, repeat_count=1):
    num_threads = multiprocessing.cpu_count()

    data_set = tf.data.TextLineDataset(filenames=file_name).skip(1)

    if shuffle:
        data_set = data_set.shuffle(buffer_size=1000)

    data_set = data_set.map(lambda row: parse_csv_row(row), num_parallel_calls=num_threads).batch(batch_size) \
        .repeat(repeat_count).prefetch(1000)

    iterator = data_set.make_one_shot_iterator()
    features, target = iterator.get_next()
    return features, target


def model_fn(features, labels, mode, params):
    is_train = False
    if mode == tf.estimator.ModeKeys.TRAIN:
        is_train = True

    vocab_table = lookup.index_table_from_file(vocabulary_file='data/vocab.csv', num_oov_buckets=1, default_value=-1)
    text = features[commons.FEATURE_COL]
    words = tf.string_split(text)
    dense_words = tf.sparse_tensor_to_dense(words, default_value=commons.PAD_WORD)
    word_ids = vocab_table.lookup(dense_words)

    padding = tf.constant([[0, 0], [0, commons.MAX_DOCUMENT_LENGTH]])
    # Pad all the word_ids entries to the maximum document length
    word_ids_padded = tf.pad(word_ids, padding)
    word_id_vector = tf.slice(word_ids_padded, [0, 0], [-1, commons.MAX_DOCUMENT_LENGTH])

    f1 = layers.embed_sequence(word_id_vector, params.N_WORDS, 50, initializer=layers.xavier_initializer(seed=42))
    f1 = tf.layers.dropout(f1, rate=0.2, training=is_train)
    f1 = tf.layers.conv1d(inputs=f1, filters=250, kernel_size=3, padding='valid',
                          kernel_initializer=layers.xavier_initializer(seed=42), activation=tf.nn.relu)
    f1 = tf.reduce_max(input_tensor=f1, axis=1)
    f1 = tf.layers.dense(f1, 256, activation=tf.nn.relu, kernel_initializer=layers.xavier_initializer(seed=42))
    f1 = tf.layers.dropout(f1, rate=0.2, training=is_train)
    f1 = tf.nn.relu(f1)
    logits = tf.layers.dense(f1, 2, kernel_initializer=layers.xavier_initializer(seed=42))

    predictions = tf.nn.softmax(logits)
    prediction_indices = tf.argmax(predictions, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        prediction_dict = {
            'class': tf.gather(commons.TARGET_LABELS, prediction_indices),
            'class_index': prediction_indices,
            'probabilities': predictions
        }

        export_outputs = {
            'predictions': tf.estimator.export.PredictOutput(prediction_dict)
        }

        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    tf.summary.scalar('loss', loss)

    acc = tf.equal(tf.cast(prediction_indices, dtype=tf.int64), tf.cast(labels, dtype=tf.int64))
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))

    tf.summary.scalar('acc', acc)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()

        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, train_op=train_op, loss=loss)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metrics_ops = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=prediction_indices)
        }
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics_ops)


def serving_fn():
    receiver_tensor = {
        commons.FEATURE_COL: tf.placeholder(dtype=tf.string, shape=None)
    }

    features = {
        key: tensor
        for key, tensor in receiver_tensor.items()
    }

    return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)
