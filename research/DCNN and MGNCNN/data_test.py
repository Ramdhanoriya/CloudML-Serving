import tensorflow as tf

from model import commons


def process(row):
    columns = tf.decode_csv(row, record_defaults=commons.HEADER_DEFAULTS, field_delim='\t')
    features = dict(zip(commons.HEADERS, columns))
    target = features.pop(commons.LABEL_COL)
    return features, tf.string_to_number(target, out_type=tf.int32)


data_set = tf.data.TextLineDataset(filenames='data/train_1.tsv')
data_set = data_set.skip(1)

data_set = data_set.map(lambda record: process(record))
data_set = data_set.batch(32)
iterator = data_set.make_initializable_iterator()

sess = tf.Session()
sess.run(iterator.initializer)
X, Y = iterator.get_next()

print(sess.run(X))
print(sess.run(Y))

X, Y = iterator.get_next()

print(sess.run(X))
print(sess.run(Y))

X, Y = iterator.get_next()

print(sess.run(X))
print(sess.run(Y))

X, Y = iterator.get_next()

print(sess.run(X))
print(sess.run(Y))


