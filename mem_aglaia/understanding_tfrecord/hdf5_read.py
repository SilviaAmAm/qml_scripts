import numpy as np
import h5py
import tensorflow as tf

hdf5_file = h5py.File("test.hdf5", "r")

mm = hdf5_file["xyz"][0]

print(mm)