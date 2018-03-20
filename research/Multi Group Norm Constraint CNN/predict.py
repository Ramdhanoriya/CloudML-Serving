__author__ = 'KKishore'

import tensorflow as tf
from tensorflow.contrib import training

from model.mgnccnn_model import model_fn

tf.logging.set_verbosity(tf.logging.INFO)

N_WORDS = 0

with open('data/nwords.csv', 'r') as f:
    N_WORDS = int(f.read()) + 2

hparams = training.HParams(
    N_WORDS=N_WORDS
)

print(N_WORDS)

estimator = tf.estimator.Estimator(model_fn=model_fn, params=hparams, model_dir='build/')


def single_instance_fn():
    return {'review': ['fuck worst bad bad bad bad worst movie']}


res = estimator.predict(input_fn=lambda: single_instance_fn())
index = 0
for i in res:
    if index == 0:
        print('{}'.format(i))
        index += 1
    else:
        break
