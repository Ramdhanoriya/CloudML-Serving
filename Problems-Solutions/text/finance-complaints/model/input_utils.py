import pandas as pd
import tensorflow as tf
import tensorflow.contrib.learn as tflearn
from tensorflow.python.platform import gfile

from model.constant import DEFAULTS, PADWORD, stop_words, MAX_FEATURES


def get_train_record(record):
    vector = tf.decode_csv(record, DEFAULTS, use_quote_delim=True)
    return vector[1:], vector[0]


def preprocess(text):
    text = text.lower()
    result = ' '.join([word for word in text.split() if word not in (stop_words)])
    return result


def build_vocab(file_name, vocab_file_name):
    df = pd.read_csv(file_name, header=None, sep=',', skiprows=[1], names=['product', 'consumer_complaint_narrative'])
    df['consumer_complaint_narrative'] = df['consumer_complaint_narrative'].apply(preprocess)
    print(df['consumer_complaint_narrative'][0])
    vocab_processor = tflearn.preprocessing.VocabularyProcessor(max_document_length=MAX_FEATURES, min_frequency=10,
                                                                tokenizer_fn=tflearn.preprocessing.tokenizer)
    vocab_processor.fit(df['consumer_complaint_narrative'])
    with gfile.Open(vocab_file_name, 'wb') as f:
        f.write("{}\n".format(PADWORD))
        for word, index in vocab_processor.vocabulary_._mapping.items():
            f.write("{}\n".format(word))
    nwords = len(vocab_processor.vocabulary_)
    print('{} words into {}'.format(nwords, vocab_file_name))


def input_fn(file_name, batch_size, repeat_count, shuffle=False):
    def _input_fn():
        data_set = tf.data.TextLineDataset(filenames=file_name)
        data_set = data_set.map(get_train_record)#data_set.skip(1).map(get_train_record)
        if shuffle:
            data_set = data_set.shuffle(shuffle)
        data_set = data_set.repeat(repeat_count)
        batch = data_set.batch(batch_size)
        iterator = batch.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return {'x': features}, labels

    return _input_fn()


def serving_input_fn():
    feature_tensor = tf.placeholder(tf.string, [None])
    # features = tf.py_func(preprocess, [feature_tensor], tf.string)
    features = tf.expand_dims(feature_tensor, -1)
    return tf.estimator.export.ServingInputReceiver({'x': features}, {'x': features})
