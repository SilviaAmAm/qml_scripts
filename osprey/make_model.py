import pickle
import numpy as np
import h5py
from qml.aglaia.aglaia import ARMP_G
from sklearn.base import clone

data = h5py.File("/Volumes/Transcend/data_sets/CN_isobutane_model/pruned_cn_isobutane/train_isopentane_cn_dft.hdf5", "r")

n_samples = 2

xyz = np.array(data.get("xyz")[:n_samples])
ene = np.array(data.get("ene")[:n_samples])*4.184
ene = ene - ene[0]
zs = np.array(data["zs"][:n_samples], dtype = int)
forces = np.array(data.get("forces")[:n_samples])*4.184

acsf_params = {"nRs2":10, "nRs3":10, "nTs": 5, "eta2":4.0, "eta3":4.0, "zeta":8.0}

estimator = ARMP_G(iterations=1, representation='acsf', representation_params=acsf_params, batch_size=2)

estimator.set_xyz(xyz)
estimator.set_classes(zs)
estimator.set_properties(ene)
estimator.set_gradients(forces)

estimator.generate_representation(method='fortran')

pickle.dump(estimator, open('model.pickle', 'wb'))

with open('idx.csv', 'w') as f:
    for i in range(n_samples):
        f.write('%s\n' % i)
