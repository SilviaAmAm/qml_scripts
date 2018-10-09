from qml.aglaia.aglaia import ARMP
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn import model_selection as modsel

acsf_params={"radial_rs":[0.8, 1.85, 2.9000000000000004, 3.95, 5.0],"angular_rs":[0.8, 1.85, 2.9000000000000004, 3.95, 5.0],
             "theta_s":[0.0, 0.7853981633974483, 1.5707963267948966, 2.356194490192345, 3.141592653589793] , 'radial_cutoff': 5,
             'angular_cutoff':5, 'zeta':17.8630, 'eta':2.5148}

# Generate estimator
estimator = ARMP(iterations=1, l1_reg=0.0001, l2_reg=0.005, learning_rate=0.0005, representation_name='acsf',
                 representation_params=acsf_params, tensorboard=False, store_frequency=10)
estimator.load_nn()

data_squal = h5py.File("/Volumes/Transcend/data_sets/CN_squalane/dft/squalane_cn_dft.hdf5", "r")

xyz_squal = np.array(data_squal.get("xyz")[:10])
zs_squal = np.array(data_squal.get("zs")[:10], dtype=np.int32)
ene_squal = np.array(data_squal.get("ene")[:10]) * 2625.50

pred1 = estimator.predict_from_xyz(xyz_squal, zs_squal)

print(pred1)