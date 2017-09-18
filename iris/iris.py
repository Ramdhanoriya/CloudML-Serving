import tensorflow as tf

import tensorflow.contrib.learn as tflearn
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.estimators import run_config

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


def experiment_fn(output_dir):
    config = run_config.RunConfig(model_dir=output_dir)
    classifier = tf.contrib.learn.DNNClassifier(hidden_units=[10, 10], feature_columns=feature_columns, n_classes=3,
                                                config=config)

    from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils

    return tflearn.Experiment(classifier,
                              train_input_fn=lambda: input_fn(train_file, perform_shuffle=True, repeat_count=10),
                              eval_input_fn=lambda: input_fn(test_file, perform_shuffle=False, repeat_count=1),
                              eval_metrics=None,
                              export_strategies=[saved_model_export_utils.make_export_strategy(
                                  serving_input_fn=iris_serving_input_fn, default_output_alternative_key=None,
                                  exports_to_keep=1
                              )],
                              train_steps=10000
                              )


learn_runner.run(experiment_fn=experiment_fn, output_dir='build/')
