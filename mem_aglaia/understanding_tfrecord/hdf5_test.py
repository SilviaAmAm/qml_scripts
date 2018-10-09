import numpy as np
import h5py
import tensorflow as tf

data = h5py.File("/Volumes/Transcend/data_sets/CN_isobutane_model/pruned_cn_isobutane/train_isopentane_cn_dft.hdf5", "r")

n_samples = 6

xyz = np.array(data.get("xyz")[:n_samples])
ene = np.array(data.get("ene")[:n_samples])*4.184
zs = np.array(data["zs"][:n_samples], dtype = int)
forces = np.array(data.get("forces")[:n_samples])*4.184

# create the new hdf5 file
hdf5_file = h5py.File("test.hdf5", mode='w')

hdf5_file.create_dataset("xyz", xyz.shape, np.float32)
hdf5_file.create_dataset("ene", ene.shape, np.float32)
hdf5_file.create_dataset("zs", zs.shape, np.int32)
hdf5_file.create_dataset("forces", forces.shape, np.float32)

# Do operations on the data and save it to file
for i in range(xyz.shape[0]):
    hdf5_file["xyz"][i] = xyz[i] * 0.5
    hdf5_file["ene"][i] = ene[i] - ene[0]
    hdf5_file["zs"][i] = zs[i]
    hdf5_file["forces"][i] = forces[i]

hdf5_file.close()
