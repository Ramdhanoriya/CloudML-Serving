import tensorflow as tf

from model.cnn_model import model_fn
from model.constant import model_dir
from model.input_utils import build_vocab, input_fn, serving_input_fn

tf.logging.set_verbosity(tf.logging.INFO)

print('Building Vocabulary.....')
build_vocab('dataset/train.csv', 'vocab.txt')

print('\n Creating Estimator')
finance_classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir)


print('\n Training .....')
finance_classifier.train(input_fn=lambda: input_fn('dataset/train.csv', batch_size=32, repeat_count=2, shuffle=True))

print('\n Evaluating.....')
eval_results = finance_classifier.evaluate(input_fn=lambda: input_fn('dataset/valid.csv', batch_size=32, repeat_count=1,
                                                                     shuffle=False))

for key in eval_results:
    print(" {} was {}".format(key, eval_results[key]))

print('\n Exporting')
exported_model_dir = finance_classifier.export_savedmodel(model_dir, serving_input_receiver_fn=serving_input_fn)
decoded_model_dir = exported_model_dir.decode("utf-8")
