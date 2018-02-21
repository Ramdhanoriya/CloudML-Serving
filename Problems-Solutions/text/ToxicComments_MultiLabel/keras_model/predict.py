from keras.models import Model
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
import pickle
from keras.preprocessing import sequence

with open('tokenizer.pkl', 'rb') as tokenizer_serializer:
    tokenizer:Tokenizer = pickle.load(tokenizer_serializer)

model: Model = load_model('fasttext.h5')

test_instance = 'Yo bitch Ja Rule is more succesful then you\'ll ever be whats up with you and hating you sad mofuckas...i should bitch slap ur pethedic white faces and get you to kiss my ass you guys sicken me. Ja rule is about pride in da music man. dont diss that shit on him. and nothin is wrong bein like tupac he was a brother too...fuckin white boys get things right next time'

x_test = tokenizer.texts_to_sequences(test_instance)
x_test = sequence.pad_sequences(x_test, maxlen=100)

print('x_test shape:', x_test.shape)

print(tokenizer.word_index)

print(model.summary())

print('\n')

print(model.predict(x=[x_test], verbose=1, steps=1))
