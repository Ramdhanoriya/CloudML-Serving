import tensorflow as tf
import numpy as np
import pickle

import json

from model.cnn_model import model_estimator
from model.input_utils import input_function, enocde_text, get_data_set

tf.logging.set_verbosity(tf.logging.INFO)

x_train, y_train = get_data_set('dataset/validpreprocess.csv')

with open('params.json', 'r') as fp:
    data = json.load(fp)

vocab_size = data['vocab_size']
max_length = data['max_length']

tokenizer=None

with open('tokenizer.p', 'rb') as fp:
    tokenizer = pickle.load(fp)

keras_estimator = model_estimator(vocab_size=vocab_size, max_length=max_length)

predict_feature, predict_target = get_data_set('dataset/validpreprocess.csv')

predict_feature = enocde_text(tokenizer, predict_feature, max_length)

predict_target = predict_target.astype(np.int32)

predict_target_cat = tf.keras.utils.to_categorical(predict_target, num_classes=11)

predict_results = keras_estimator.predict(input_fn=input_function(predict_feature, labels=predict_target_cat, shuffle=False, batch_size=32, epochs=1))

predictions = list(predict_results)
predicted_values = []
for p in predictions:
    class_val = np.argmax(p['output'])
    print('Predicted = ', class_val, p)
    predicted_values.append(class_val)

from sklearn.metrics import accuracy_score

print('Predicted = ', predicted_values)
print('Actual =', predict_target)

print('Accuracy = ', accuracy_score(y_pred=predicted_values, y_true=predict_target))
