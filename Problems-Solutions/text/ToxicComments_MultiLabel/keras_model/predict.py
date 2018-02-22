from keras.models import Model
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
import pickle
from keras.preprocessing import sequence
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

text_col = 'comment_text'
target_col = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

with open('tokenizer.pkl', 'rb') as tokenizer_serializer:
    tokenizer:Tokenizer = pickle.load(tokenizer_serializer)

model: Model = load_model('fasttext.h5')
print(model.summary())


test_df = pd.read_csv("data/test_m.csv").fillna("sterby")
actual_values = y_train = test_df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
predicted_values = []

print(test_df.size)

for index, row in test_df.iterrows():
    print('For Index = ', index)
    test_instance = row[text_col]
    x_test = tokenizer.texts_to_sequences(test_instance)
    x_test = sequence.pad_sequences(x_test, maxlen=100)
    predicted_instance = model.predict(x=x_test, verbose=1, steps=1)
    probs = predicted_instance[0]
    predicted_label = []
    for item in probs:
        if item > 0.30:
            predicted_label.append(1)
        else:
            predicted_label.append(0)
    predicted_values.append(predicted_label)

predicted_values = np.array(predicted_values, dtype=np.int32)
actual_values = np.array(actual_values, dtype=np.int32)

print(predicted_values)
print(actual_values)

print(accuracy_score(actual_values, predicted_values))

