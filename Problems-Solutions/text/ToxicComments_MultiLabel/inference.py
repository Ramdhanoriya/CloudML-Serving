__author__ = 'KKishore'

import tensorflow as tf

from tensorflow.contrib import predictor

tf.logging.set_verbosity(tf.logging.INFO)

base_dir = 'serving/1519062510'

prediction_fn = predictor.from_saved_model(export_dir=base_dir, signature_def_key='predictions')

output = prediction_fn({
    'comment_text': [
        'thanks for the comment ya wikipedia dickhead ! ! !'
    ]
})

print(output['class'])
import numpy as np
print(np.argmax(output['class'], axis=-1))