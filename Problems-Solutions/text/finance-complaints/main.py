import tensorflow as tf
import numpy as np

import json

from model.cnn_model import model_estimator
from model.input_utils import input_function, enocde_text, get_data_set

tf.logging.set_verbosity(tf.logging.INFO)

x_train, y_train = get_data_set('dataset/trainpreprocess.csv')
x_test, y_test = get_data_set('dataset/testpreprocess.csv')

print('Train Data = ', np.shape(x_train))
print('Test Data = ', np.shape(x_test))

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=1000)
tokenizer.fit_on_texts(x_train)

try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle

with open('tokenizer.p', 'wb') as fp:
    pickle.dump(tokenizer, fp, protocol=pickle.HIGHEST_PROTOCOL)

vocab_size = len(tokenizer.word_index) + 1
max_length = max([len(s.split()) for s in x_train])

print('Vocab Size = ', vocab_size)
print('Max Length = ', max_length)

params = {}
params['vocab_size'] = vocab_size
params['max_length'] = max_length

with open('params.json', 'w') as fp:
    json.dump(params, fp)

x_train = enocde_text(tokenizer, x_train, max_length)
x_test = enocde_text(tokenizer, x_test, max_length)

y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=11)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=11)

keras_estimator = model_estimator(vocab_size, max_length)

print('Training.....')
keras_estimator.train(input_fn=input_function(x_train, y_train, shuffle=True, batch_size=37, epochs=10))

print('Evaluating......')
eval_results = keras_estimator.evaluate(input_fn=input_function(x_test, y_test, shuffle=False, batch_size=37, epochs=1))

print('Evaluated Results - '.format(eval_results))
