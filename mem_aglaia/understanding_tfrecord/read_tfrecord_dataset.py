import numpy as np
import h5py
import tensorflow as tf

def _read_from_tfrecord(example_proto):
    feature = {
        #  'n_atoms': tf.FixedLenFeature([], tf.int64),
        # 'n_space': tf.FixedLenFeature([], tf.int64),
        # 'n_feat': tf.FixedLenFeature([], tf.int64),
        'g_raw': tf.FixedLenFeature([], tf.string),
        'dg_raw': tf.FixedLenFeature([], tf.string),
        'ene_raw': tf.FixedLenFeature([], tf.string),
        'zs_raw': tf.FixedLenFeature([], tf.string),
        'forces_raw': tf.FixedLenFeature([], tf.string),
    }

    features = tf.parse_example([example_proto], features=feature)

    g_1d = tf.decode_raw(features['g_raw'], tf.float64)
    dg_1d = tf.decode_raw(features['dg_raw'], tf.float64)
    ene_1d = tf.decode_raw(features['ene_raw'], tf.float64)
    zs_1d = tf.decode_raw(features['zs_raw'], tf.float64)
    forces_1d = tf.decode_raw(features['forces_raw'], tf.float64)

    g = tf.cast(tf.reshape(g_1d, tf.stack([19, 75])), tf.float32)
    ene = tf.cast(tf.reshape(ene_1d, tf.stack([1, ])), tf.float32)
    dg = tf.cast(tf.reshape(dg_1d, tf.stack([19, 75, 19, 3])), tf.float32)
    zs = tf.cast(tf.reshape(zs_1d, tf.stack([19, ])), tf.int32)
    forces = tf.cast(tf.reshape(forces_1d, tf.stack([19, 3])), tf.float32)

    return g, dg, ene, forces, zs


data_path = 'predict.tfrecords'

dataset = tf.data.TFRecordDataset(data_path)
dataset = dataset.map(_read_from_tfrecord)

iterator = dataset.make_initializable_iterator()
g, dg, ene, forces, zs = iterator.get_next()

sess = tf.Session()

iterator_init = iterator.make_initializer(dataset)
init = tf.global_variables_initializer()

sess.run([iterator.initializer, init])
print(sess.run(g))