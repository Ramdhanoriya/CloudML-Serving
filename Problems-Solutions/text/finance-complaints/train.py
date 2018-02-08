import tensorflow as tf
from tensorflow.contrib import training
from model.cnn_model_v2 import model_fn, input_fn

tf.logging.set_verbosity(tf.logging.INFO)

N_WORDS = 0

with open('dataset/nwords.csv', 'r') as f:
    N_WORDS = int(f.read()) + 2

hparams = training.HParams(
    N_WORDS=N_WORDS
)

estimator = tf.estimator.Estimator(model_fn=model_fn, params=hparams, model_dir='build2/')

estimator.train(input_fn=lambda: input_fn('dataset/trainpreprocess.csv', shuffle=True, repeat_count=5))

evaluated_results = estimator.evaluate(input_fn=lambda: input_fn('dataset/testpreprocess.csv', shuffle=False, repeat_count=1))

print("# Evaluated Results: {}".format(evaluated_results))

#estimator.export_savedmodel(export_dir_base='build2', serving_input_receiver_fn=serving_fn, as_text=True)
