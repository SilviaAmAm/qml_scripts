import numpy as np
import h5py
import tensorflow as tf

def _int64_feature(value_dim):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value_dim]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

data = h5py.File("/Volumes/Transcend/data_sets/CN_isobutane_model/pruned_cn_isobutane/train_isopentane_cn_dft.hdf5", "r")

n_samples = 6

xyz = np.array(data.get("xyz")[:n_samples])
ene = np.array(data.get("ene")[:n_samples])*4.184
zs = np.array(data["zs"][:n_samples], dtype = int)
forces = np.array(data.get("forces")[:n_samples])*4.184

# create the new file

writer = tf.python_io.TFRecordWriter("test.tfrecords")

for i in range(xyz.shape[0]):
    xyz_sample = xyz[i]
    zs_sample = zs[i]

    # Storing the shape
    dim_0, dim_1 = xyz_sample.shape

    # Creating god damned examples
    example = tf.train.Example(features=tf.train.Features(feature={
        'n_atoms': _int64_feature(dim_0),
        'n_space': _int64_feature(dim_1),
        'n_feat': _int64_feature(dim_1),
        'xyz_raw': _bytes_feature(tf.compat.as_bytes(xyz_sample.tostring())),
        'zs_raw': _bytes_feature(tf.compat.as_bytes(zs_sample.tostring()))
    }))

    # Writing to record
    writer.write(example.SerializeToString())

writer.close()

