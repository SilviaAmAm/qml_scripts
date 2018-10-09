
import numpy as np
from qml.aglaia.aglaia import ARMP_G
import h5py

data = h5py.File("/Volumes/Transcend/data_sets/CN_isobutane_model/pruned_cn_isobutane/train_isopentane_cn_dft.hdf5", "r")

n_samples = 10

xyz = np.array(data.get("xyz")[:n_samples])
ene = np.array(data.get("ene")[:n_samples])*4.184
zs = np.array(data["zs"][:n_samples], dtype=np.int32)
forces = np.array(data.get("forces")[:n_samples])*4.184

## ------------- ** Setting up the estimator ** ---------------

estimator = ARMP_G(iterations=10, representation_name='acsf', representation_params={"nRs2": 5, "nRs3": 5,
"nTs": 2}, tensorboard=False, store_frequency=1, batch_size=20)

estimator.set_xyz(xyz[:n_samples])
estimator.set_classes(zs[:n_samples])
estimator.set_properties(ene[:n_samples])
estimator.set_gradients(forces[:n_samples])

idx = np.arange(0, n_samples)

estimator.fit(idx)
print("\n Done the fitting")

estimator.save_nn()

ene_1, f_1 = estimator.predict(idx)



