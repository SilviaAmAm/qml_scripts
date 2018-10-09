import numpy as np
import h5py
import tensorflow as tf

data_path = 'test.tfrecords'

feature = {'n_atoms': tf.FixedLenFeature([], tf.int64),
            'n_space': tf.FixedLenFeature([], tf.int64),
            'xyz_raw': tf.FixedLenFeature([], tf.string),
            'zs_raw': tf.FixedLenFeature([], tf.string)
           }

filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)

features = tf.parse_single_example(serialized_example, features=feature)

n_atoms = features['n_atoms']
n_space = features['n_space']
xyz_1d = tf.decode_raw(features['xyz_raw'], tf.float64)
zs_1d = tf.decode_raw(features['zs_raw'], tf.int64)

xyz = tf.reshape(xyz_1d, tf.stack([n_atoms, n_space]))
zs = tf.reshape(zs_1d, tf.stack([n_atoms]))

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

with tf.Session()  as sess:
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(1):
        sample_xyz = sess.run(xyz)
        sample_zs = sess.run(zs)

        print(sample_xyz)
        print(sample_zs)