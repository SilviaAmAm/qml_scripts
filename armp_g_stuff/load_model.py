from qml.aglaia.aglaia import ARMP_G
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn import model_selection as modsel

data = h5py.File("/Volumes/Transcend/data_sets/CN_isobutane_model/pruned_cn_isobutane/train_isopentane_cn_dft.hdf5", "r")

n_samples = 12

idx = list(range(n_samples))
splitter = modsel.KFold(n_splits=3, random_state=42, shuffle=True)
indices = splitter.split(idx)

xyz = np.array(data.get("xyz")[:n_samples])
ene = np.array(data.get("ene")[:n_samples])*4.184
zs = np.array(data["zs"][:n_samples])
forces = np.array(data.get("forces")[:n_samples])*4.184

estimator = ARMP_G(representation_params={"nRs2": 5, "nRs3": 5,"nTs": 2})

estimator.load_nn("saved_model")

estimator.set_xyz(xyz[:n_samples])
estimator.set_classes(zs[:n_samples])
estimator.set_properties(ene[:n_samples])
estimator.set_gradients(forces[:n_samples])

# estimator.load_representations_and_dgdr("descrpt_and_grad.hdf5")

idx = np.arange(0, n_samples)
# estimator.fit(idx)

ene_1, f_1 = estimator.predict(idx)

# for item in indices:
#     idx_test = item[1]
#     ene_1, f_1 = estimator.predict(idx_test)
#
#     plt.scatter(ene_1, ene[:idx_test])
#     plt.show()

# ene_2, f_2 = estimator.predict_from_xyz(xyz[:n_samples], zs[:n_samples])


#
# plt.scatter(ene_1, ene_2)
# plt.show()

