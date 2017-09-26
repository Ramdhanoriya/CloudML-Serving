import time

import tensorflow as tf
import tensorflow.contrib.learn as tflearn
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec

print('Tensorflow Version - ', tf.__version__)  # Tensorflow 1.3
tf.logging.set_verbosity(tf.logging.INFO)

train_file = 'dataset/iris_training.csv'
test_file = 'dataset/iris_test.csv'

feature_names = [
    'SepalLength',
    'SepalWidth',
    'PetalLength',
    'PetalWidth'
]


def input_fn(file, perform_shuffle=False, repeat_count=1):
    def decode_csv(line):
        parsed_line = tf.decode_csv(line, [[0.], [0.], [0.], [0.], [0]])
        label = parsed_line[-1:]
        del parsed_line[-1]
        features = parsed_line
        parsed_data = dict(zip(feature_names, features)), label
        return parsed_data

    data_set = (tf.contrib.data.TextLineDataset(file).skip(1).map(decode_csv))

    if perform_shuffle:
        data_set = data_set.shuffle(buffer_size=256)

    data_set = data_set.repeat(repeat_count)
    data_set = data_set.batch(32)
    iterator = data_set.make_one_shot_iterator()
    batch_features, batch_label = iterator.get_next()
    return batch_features, batch_label


feature_columns = [tf.feature_column.numeric_column(feature) for feature in feature_names]


def iris_serving_input_fn():
    """Build the serving inputs."""

    inputs = {}
    for feat in feature_columns:
        inputs[feat.name] = tf.placeholder(shape=[None], dtype=feat.dtype)

    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in inputs.items()
    }
    return tf.contrib.learn.InputFnOps(features, None, inputs)


validation_metrics = {
    "accuracy": MetricSpec(metric_fn=tf.contrib.metrics.streaming_accuracy, prediction_key="classes"),
    "recall": MetricSpec(metric_fn=tf.contrib.metrics.streaming_recall, prediction_key="classes"),
    "precision": MetricSpec(metric_fn=tf.contrib.metrics.streaming_precision, prediction_key="classes")
}
validation_monitor = tflearn.monitors.ValidationMonitor(input_fn=lambda: input_fn(test_file, perform_shuffle=False,
                                                                                  repeat_count=1), every_n_steps=50,
                                                        metrics=validation_metrics, early_stopping_metric="loss",
                                                        early_stopping_metric_minimize=True, early_stopping_rounds=200)

classifier = tflearn.DNNClassifier(hidden_units=[10, 10], feature_columns=feature_columns, n_classes=3,
                                   model_dir='build/', config=tflearn.RunConfig(save_checkpoints_secs=1))

classifier.fit(input_fn=lambda: input_fn(train_file, perform_shuffle=True, repeat_count=40),
               monitors=[validation_monitor])

evaluation_results = classifier.evaluate(input_fn=lambda: input_fn(test_file, perform_shuffle=False, repeat_count=1))

for key in evaluation_results:
    print(" {} was {}".format(key, evaluation_results[key]))

time.sleep(5)
print('\n\n Exporting Iris Model')

classifier.export_savedmodel(export_dir_base='build/', serving_input_fn=iris_serving_input_fn,
                             default_output_alternative_key=None)
