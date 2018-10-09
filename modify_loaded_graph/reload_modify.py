import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np

feed_a = np.ones(shape=(2, 4))
new_feed_a = np.ones(shape=(2, 6))

new_a = tf.placeholder(shape=[2, 6], dtype=tf.float32, name="Input_a_new")
new_b = tf.ones(shape=[6, 3], name='b_new')
new_m = tf.matmul(new_a, new_b, name="matmul_new")

with tf.Session() as sess:

    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], "saved_model", input_map={"matmul:0": new_m})

    new_out = tf.get_default_graph().get_tensor_by_name("out:0")
    new_result = sess.run(new_out, feed_dict={new_a: new_feed_a})

print(new_result)
