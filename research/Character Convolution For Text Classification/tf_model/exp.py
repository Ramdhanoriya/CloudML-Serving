import tensorflow as tf
from tensorflow.contrib import lookup


text = ['this is awesome']

split_word = tf.string_split(text)
x = split_word.values
split_chars = tf.string_split(x, delimiter='')
#dense_words = tf.sparse_tensor_to_dense(split_chars, default_value='$K#')
rev = tf.reverse(split_chars.values, axis=[-1])


table = lookup.index_table_from_file(vocabulary_file='v.txt', vocab_size=69, default_value=0)

#dense_words = tf.sparse_tensor_to_dense(split_chars, default_value='#')
word_ids = table.lookup(rev)
word_ids = tf.expand_dims(word_ids, axis=[0])

padding = tf.constant([[0, 0], [0, 1014]])
# Pad all the word_ids entries to the maximum document length
word_ids_padded = tf.pad(word_ids, padding)
word_id_vector = tf.slice(word_ids_padded, [0, 0], [-1, 1014])

shape = tf.shape(word_id_vector)

size = tf.keras.backend.ndim(word_id_vector)
print(tf.keras.backend.shape(word_id_vector))

print(size)

sess = tf.Session()

sess.run(tf.tables_initializer())

print(sess.run(rev))

print(sess.run(word_id_vector))

print(sess.run(shape))


