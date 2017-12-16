import tensorflow as tf
from tensorflow.contrib import lookup

from model.constant import PADWORD, MAX_FEATURES, MAX_LEN, filters, kernel_size, hidden_dims


def model_fn(features, labels, mode):
    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.keras.backend.set_learning_phase(True)
    else:
        tf.keras.backend.set_learning_phase(False)

    input_feature = features['x']
    table = lookup.index_table_from_file(vocabulary_file='vocab.txt', num_oov_buckets=1, default_value=-1)
    text = tf.squeeze(input_feature, [1])
    words = tf.string_split(text)
    dense_words = tf.sparse_tensor_to_dense(words, default_value=PADWORD)
    numbers = table.lookup(dense_words)
    padding = tf.constant([[0, 0], [0, MAX_LEN]])
    padded = tf.pad(numbers, padding)
    sliced = tf.slice(padded, [0, 0], [-1, MAX_LEN])
    print('words_sliced={}'.format(words))

    embeds = tf.keras.layers.Embedding(MAX_FEATURES, 128, input_length=MAX_LEN)(sliced)

    print('words_embed={}'.format(embeds))

    f1 = tf.keras.layers.Dropout(0.2)(embeds)
    f1 = tf.keras.layers.Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1)(f1)
    f1 = tf.keras.layers.GlobalAveragePooling1D()(f1)
    f1 = tf.keras.layers.Dense(hidden_dims)(f1)
    f1 = tf.keras.layers.Dropout(0.5)(f1)
    f1 = tf.keras.layers.Activation('relu')(f1)
    logits = tf.keras.layers.Dense(11)(f1)

    predictions_dict = {
        'class': tf.argmax(logits, 1),
        'prob': tf.nn.softmax(logits)
    }

    '''prediction_output = tf.estimator.export.PredictOutput({"classes": tf.argmax(input=logits, axis=1),
                                                           "probabilities": tf.nn.softmax(logits,
                                                                                          name="softmax_tensor")})'''

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions_dict, export_outputs={
            'prediction': tf.estimator.export.PredictOutput(predictions_dict)
        })

    # one_hot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=11)
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits=logits)

    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(loss, tf.contrib.framework.get_global_step(), optimizer='Adam',
                                                   learning_rate=0.001)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metrics_ops = {
        'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions_dict['class']),
        'precision': tf.metrics.precision(labels=labels, predictions=predictions_dict['class']),
        'recall': tf.metrics.recall(labels=labels, predictions=predictions_dict['class'])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics_ops)
