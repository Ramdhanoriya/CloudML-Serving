import tensorflow as tf
from tensorflow.contrib import lookup, layers
import multiprocessing
from model import commons


def parse_csv_row(row):
    columns = tf.decode_csv(row, record_defaults=commons.HEADER_DEFAULTS, field_delim='\t')
    features = dict(zip(commons.HEADERS, columns))
    target = features.pop(commons.LABEL_COL)

    features[commons.WEIGHT_COLUNM_NAME] = tf.cond(
        tf.equal(target, commons.TARGET_LABELS[0]),
        lambda: 6.2,
        lambda: 1.0
    )
    return features, target


def decode_label(label_string):
    table = lookup.index_table_from_tensor(tf.constant(commons.TARGET_LABELS))
    return table.lookup(label_string)


def input_fn(file_name, batch_size=16, shuffle=False, repeat_count=1):
    num_threads = multiprocessing.cpu_count()

    data_set = tf.data.TextLineDataset(filenames=file_name).skip(1)

    if shuffle:
        data_set = data_set.shuffle(buffer_size=1000)

    data_set = data_set.map(lambda row: parse_csv_row(row), num_parallel_calls=num_threads).batch(batch_size) \
        .repeat(repeat_count).prefetch(1000)

    iterator = data_set.make_one_shot_iterator()
    features, target = iterator.get_next()
    return features, decode_label(target)


def model_fn(features, labels, mode, params):
    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.keras.backend.set_learning_phase(True)
    else:
        tf.keras.backend.set_learning_phase(False)

    vocab_table = lookup.index_table_from_file(vocabulary_file='data/vocab.csv', num_oov_buckets=1, default_value=-1)
    text = features[commons.FEATURE_COL]
    words = tf.string_split(text)
    dense_words = tf.sparse_tensor_to_dense(words, default_value=commons.PAD_WORD)
    word_ids = vocab_table.lookup(dense_words)

    padding = tf.constant([[0, 0], [0, commons.MAX_DOCUMENT_LENGTH]])
    # Pad all the word_ids entries to the maximum document length
    word_ids_padded = tf.pad(word_ids, padding)
    word_id_vector = tf.slice(word_ids_padded, [0, 0], [-1, commons.MAX_DOCUMENT_LENGTH])

    word_embeddings = layers.embed_sequence(word_id_vector, vocab_size=params.N_WORDS, embed_dim=50)

    min_vectors = tf.reduce_min(word_embeddings, axis=1)
    max_vectors = tf.reduce_max(word_embeddings, axis=1)

    min_max_vectors = tf.concat([min_vectors, max_vectors], axis=1)

    d1 = tf.keras.layers.Dense(25, activation='relu')(min_max_vectors)
    logits = tf.keras.layers.Dense(commons.TARGET_SIZE)(d1)

    probabilities = tf.nn.softmax(logits)
    predicted_indices = tf.argmax(probabilities, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class': tf.gather(commons.TARGET_LABELS, predicted_indices),
            'probabilities': probabilities
        }

        exported_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=exported_outputs)

    weights = features[commons.WEIGHT_COLUNM_NAME]

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, weights=weights)
    tf.summary.scalar('loss', loss)

    acc = tf.equal(predicted_indices, labels)
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))

    tf.summary.scalar('acc', acc)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metrics_ops = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=predicted_indices, weights=weights)
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
