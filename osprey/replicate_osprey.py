import numpy as np
import h5py
from qml.aglaia.aglaia import ARMP_G
from sklearn import model_selection as modsel
import tensorflow as tf


data = h5py.File("/Volumes/Transcend/data_sets/CN_isobutane_model/pruned_cn_isobutane/train_isopentane_cn_dft.hdf5", "r")

n_samples = 6

xyz = np.array(data.get("xyz")[:n_samples])
ene = np.array(data.get("ene")[:n_samples])*4.184
ene = ene - ene[0]
zs = np.array(data["zs"][:n_samples], dtype = int)
forces = np.array(data.get("forces")[:n_samples])*4.184

acsf_params = {"radial_rs":np.arange(0,10, 0.6), "angular_rs":np.arange(0.5, 10.5, 0.6), "theta_s": np.arange(0, 3.14, 0.5), "eta":4.0, "zeta":8.0}

idx = list(range(n_samples))

splitter = modsel.KFold(n_splits=3, random_state=42, shuffle=True)
indices = splitter.split(idx)

all_scores = []

for item in indices:

    estimator = ARMP_G(iterations=1, representation='acsf', representation_params=acsf_params, batch_size=250, tensorboard=True)

    estimator.set_xyz(xyz)
    estimator.set_classes(zs)
    estimator.set_properties(ene)
    estimator.set_gradients(forces)

    estimator.generate_representation(method='fortran')

    idx_train = item[0]
    idx_test = item[1]

    estimator.fit(idx_train)
    score = estimator.score(idx_test)
    all_scores.append(score)
    tf.reset_default_graph()


print(all_scores)



