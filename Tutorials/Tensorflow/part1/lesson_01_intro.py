import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

hello = tf.constant('Hello World')

sess = tf.Session()

print(sess.run(hello))

a = tf.constant(10, dtype=tf.float32, name='A')
b = tf.constant(22, dtype=tf.float32, name='B')

c = tf.add(a, b, name='C')

print('Addition of tow numbers - ', sess.run(c))

alpha = tf.placeholder(dtype=tf.float32, shape=None)
beta = tf.placeholder(dtype=tf.float32, shape=None)

gamma = alpha + beta

print('\n Feed Dict')

print('Sum of 100, 200  = {}'.format(sess.run(gamma, feed_dict={alpha:100, beta:200})))
print('Sum of 1050, 2050 = {}'.format(sess.run(gamma, feed_dict={alpha:1050, beta:2050})))

sess.close()