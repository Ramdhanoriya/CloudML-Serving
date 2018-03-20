__author__ = 'KKishore'

import tensorflow as tf

from tensorflow.contrib import predictor

tf.logging.set_verbosity(tf.logging.INFO)

base_dir = 'serving/1521565629'

prediction_fn = predictor.from_saved_model(export_dir=base_dir, signature_def_key='serving_default')

output = prediction_fn({
    'inputs': [
        'this movie is so cool and best and beautiful'
    ]
})

print(output)
