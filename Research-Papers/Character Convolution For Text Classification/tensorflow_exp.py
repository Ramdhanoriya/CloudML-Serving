import tensorflow as tf
from tensorflow.contrib import lookup

text = ['this is awesome']

split_word = tf.string_split(text)
x = split_word.values
split_chars = tf.string_split(x, delimiter='')

table = lookup.index_table_from_file(vocabulary_file='data/vocab.csv', vocab_size=69, default_value=0)

encoded = tf.one_hot(table.lookup(split_chars.values), 9, dtype=tf.float32)

encoded = tf.expand_dims(encoded, axis=1)

sess = tf.Session()

sess.run(tf.tables_initializer())

print(sess.run(encoded))
