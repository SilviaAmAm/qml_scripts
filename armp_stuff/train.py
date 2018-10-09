from qml.aglaia.aglaia import ARMP
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn import model_selection as modsel
import tensorflow as tf

# Getting the dataset
data = h5py.File("/Volumes/Transcend/data_sets/CN_isopentane/pruned_dft_with_forces/pruned_isopentane_cn_dft.hdf5", "r")

n_samples = 500

xyz = np.array(data.get("xyz")[-n_samples:])
ene = np.array(data.get("ene")[-n_samples:])*2625.50
ene = ene - data.get("ene")[0]*2625.50
zs = np.array(data["zs"][-n_samples:], dtype=np.int32)

# Creating the estimator
acsf_param = {"nRs2": 5, "nRs3": 5, "nTs": 5, "rcut": 5, "acut": 5, "zeta": 220.127, "eta": 30.8065}
estimator = ARMP(iterations=1000, batch_size=512, l1_reg=0.0, l2_reg=0.0, learning_rate=0.001, representation_name='acsf',
                 representation_params=acsf_param, tensorboard=False, store_frequency=50)
estimator.set_properties(ene)
estimator.generate_representation(xyz, zs, method='fortran')
print(estimator.g.shape)

# Doing cross validation
idx = list(range(n_samples))
idx_train, idx_test = modsel.train_test_split(idx, test_size=0.15, random_state=42, shuffle=False)

print("Starting the fitting...")
estimator.fit(idx_train)

# estimator.save_nn("saved_model")

pred1 = estimator.predict(idx_train)
pred2 = estimator.predict_from_xyz(xyz[idx_train], zs[idx_train])


