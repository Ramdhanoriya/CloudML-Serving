import tensorflow as tf

from model.cnn_model import model_fn
from model.constant import model_dir
from model.input_utils import input_fn

tf.logging.set_verbosity(tf.logging.INFO)

# print('Building Vocabulary.....')
# build_vocab('dataset/train.csv', 'vocab.txt')

print('\n Creating Estimator')
finance_classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir)

predict_fn = lambda: input_fn('dataset/predict.csv', batch_size=32, repeat_count=1, shuffle=False)

predictions = list(finance_classifier.predict(input_fn=predict_fn))

print("")

print("* Predicted Classes: {}".format(list(map(lambda item: item["class"], predictions))))

print("* Predicted Probabilities: {}".format(list(map(lambda item: list(item["prob"]), predictions))))
