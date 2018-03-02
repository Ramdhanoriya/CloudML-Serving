import tensorflow as tf
from tensorflow.contrib import lookup

text = ['this is awesome']

split_word = tf.string_split(text)
x = split_word.values
split_chars = tf.string_split(x, delimiter='')

table = lookup.index_table_from_file(vocabulary_file='data/vocab.csv', vocab_size=69, default_value=0)

dense_words = tf.sparse_tensor_to_dense(split_chars, default_value='#')
word_ids = table.lookup(dense_words)

padding = tf.constant([[0, 0], [0, 100]])
# Pad all the word_ids entries to the maximum document length
word_ids_padded = tf.pad(word_ids, padding)
word_id_vector = tf.slice(word_ids_padded, [0, 0], [-1, 100])

size = tf.keras.backend.ndim(word_id_vector)
print(tf.keras.backend.shape(word_id_vector))

print(size)

'''
encoded = tf.one_hot(table.lookup(split_chars.values), 9, dtype=tf.float32)

encoded = tf.expand_dims(encoded, axis=1)
'''

sess = tf.Session()

sess.run(tf.tables_initializer())

print(sess.run(word_id_vector))
