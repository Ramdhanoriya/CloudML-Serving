import tensorflow as tf
import numpy as np

from model.cnn_model import model_estimator
from model.input_utils import input_function, enocde_text, get_data_set

tf.logging.set_verbosity(tf.logging.INFO)

x_train, y_train = get_data_set('dataset/validpreprocess.csv')