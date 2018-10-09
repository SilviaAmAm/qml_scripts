from qml.aglaia.aglaia import ARMP_G
import h5py
import numpy as np
import matplotlib.pyplot as plt

data = h5py.File("/Volumes/Transcend/data_sets/CN_isobutane_model/pruned_dft_with_forces/pruned_isopentane_cn_dft.hdf5", "r")

n_samples = 100

xyz = np.array(data.get("xyz")[-n_samples:])
ene = np.array(data.get("ene")[-n_samples:])*2625.50
ene = ene - data.get("ene")[0]*2625.50
zs = np.array(data["zs"][-n_samples:], dtype=np.int32)
forces = np.array(data.get("forces")[-n_samples:])

idx = list(range(n_samples))

# Creating the estimator
estimator = ARMP_G(iterations=50, batch_size=10)

estimator.set_xyz(xyz)
estimator.set_properties(ene)
estimator.set_classes(zs)
estimator.set_gradients(forces)

estimator.load_nn()

estimator.predict(idx)

estimator.fit(idx)

estimator.score(idx)