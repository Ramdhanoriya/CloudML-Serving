conv_layers = conv_layers = [[256, 7, 3],
                             [256, 7, 3],
                             [256, 3, None],
                             [256, 3, None],
                             [256, 3, None],
                             [256, 3, 3]]
fully_layers = [1024, 1024]
l0 = 1014
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
alphabet_size = len(alphabet)

embedding_size = 128
num_of_classes = 4
th = 1e-6
dropout_p = 0.5

base_rate = 1e-2
momentum = 0.9
decay_step = 15000
decay_rate = 0.95
epochs = 5000
evaluate_every = 100
checkpoint_every = 100
batch_size = 128

from keras_model.data_utils import Data
import numpy as np

train_data = Data(data_source='data/ag_news_csv/train.csv', alphabet=alphabet, l0=l0, batch_size=0,
                  no_of_classes=num_of_classes)

train_data.loadData()

X_train, y_train = train_data.getAllData()

print(np.shape(X_train))
print(X_train)