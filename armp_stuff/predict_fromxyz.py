from qml.aglaia.aglaia import ARMP
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn import model_selection as modsel
import tensorflow as tf

# Getting the dataset
data = h5py.File("/Volumes/Transcend/data_sets/CN_isobutane_model/pruned_dft_with_forces/pruned_isopentane_cn_dft.hdf5", "r")

n_samples = 50

xyz = np.array(data.get("xyz")[-n_samples:])
ene = np.array(data.get("ene")[-n_samples:])*2625.50
ene = ene - data.get("ene")[0]*2625.50
zs = np.array(data["zs"][-n_samples:], dtype=np.int32)

# Creating the estimator
acsf_params = {"nRs2": 5, "nRs3": 5, "nTs": 5, "rcut": 5, "acut": 5, "zeta": 220.127, "eta": 30.8065}
estimator = ARMP(iterations=200, representation_name='acsf', representation_params=acsf_params, tensorboard=True,
                 store_frequency=10, l1_reg=0.0001, l2_reg=0.005, learning_rate=0.0005)

estimator.set_properties(ene)
estimator.generate_representation(xyz, zs)

saved_dir = "saved_model"

estimator.load_nn(saved_dir)

idx = list(range(n_samples))

estimator.fit(idx)

pred_1 = estimator.predict(idx)
pred_2 = estimator.predict_from_xyz(xyz, zs)


# plt.scatter(pred_1, pred_2)
# plt.show()

# estimator.save_nn()
