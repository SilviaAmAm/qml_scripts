from qml.aglaia.aglaia import ARMP
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn import model_selection as modsel
import tensorflow as tf

# Getting the dataset
data = h5py.File("/Volumes/Transcend/data_sets/CN_isobutane_model/pruned_dft_with_forces/pruned_isopentane_cn_dft.hdf5", "r")

n_samples = 300

xyz = np.array(data.get("xyz")[-n_samples:])
ene = np.array(data.get("ene")[-n_samples:])*2625.50
ene = ene - data.get("ene")[0]*2625.50
zs = np.array(data["zs"][-n_samples:], dtype=np.int32)

# Creating the estimator
estimator = ARMP(iterations=100, l1_reg=0.0001, l2_reg=0.005, learning_rate=0.0005, representation='acsf',
                 representation_params={"radial_rs": np.arange(0, 10, 3),"angular_rs": np.arange(0, 10, 3),
                                        "theta_s": np.arange(0, 3.14, 3)}, tensorboard=True, store_frequency=10, tensorboard_subdir="tb")

estimator.set_properties(ene)
estimator.generate_representation(xyz, zs)

saved_dir = "saved_model"

estimator.load_nn(saved_dir)

idx = list(range(n_samples))
idx_train, idx_test = modsel.train_test_split(idx, random_state=42, shuffle=True)

pred3 = estimator.predict(idx)

estimator.fit(idx_train)

print(estimator.predict(idx_test))
# estimator.score(idx_test)
