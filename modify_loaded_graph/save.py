import tensorflow as tf
import numpy as np

feed_a = np.ones(shape=(2, 4))

a = tf.placeholder(shape=[2, 4], dtype=tf.float32, name="Input_a")
b = tf.ones(shape=[4, 3], name='b')

m = tf.matmul(a, b, name="matmul")

c = tf.ones(shape=[2, 3], name="c")

out = tf.add(m, c, name="out")

with tf.Session() as sess:
    result = sess.run(out, feed_dict={a: feed_a})
    print(result)

    graph = tf.get_default_graph()

    a_tf = graph.get_tensor_by_name("Input_a:0")
    out_tf = graph.get_tensor_by_name("out:0")
    tf.saved_model.simple_save(sess, export_dir="saved_model", inputs={"Input_a:0": a_tf}, outputs={"out:0":out_tf})




