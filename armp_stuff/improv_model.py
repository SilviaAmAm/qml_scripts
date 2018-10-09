import tensorflow as tf
import numpy as np

np.random.seed(0)

elements = [1, 6, 7]

zs = np.array([[6, 1, 1, 1], [6, 7, 1, 0]])
x = np.random.randint(low=0, high=10, size=(2, 4, 8))

zs_tf = tf.placeholder(shape=[None, 4], dtype=tf.int32)
x_tf = tf.placeholder(shape=[None, 4, 8], dtype=tf.int32)

all_atomic_energies = tf.zeros_like(zs, dtype=tf.int32)

for el in elements:
    current_element = tf.expand_dims(tf.constant(el, dtype=tf.int32), axis=0)
    where_element = tf.cast(tf.where(tf.equal(zs_tf, current_element)), dtype=tf.int32)

    current_element_in_x = tf.gather_nd(x_tf, where_element)

    atomic_ene = tf.reduce_sum(current_element_in_x, axis=-1)

    updates = tf.scatter_nd(where_element, atomic_ene, tf.shape(zs_tf))
    all_atomic_energies = tf.add(all_atomic_energies, updates)

    with tf.Session() as sess:
        result = sess.run(all_atomic_energies, feed_dict={x_tf: x, zs_tf:zs})
        print(result)


