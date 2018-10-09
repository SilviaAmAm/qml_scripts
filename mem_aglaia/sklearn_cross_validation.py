import tensorflow as tf
from sklearn import model_selection as modsel
import os
import numpy as np
import h5py
from qml.aglaia.aglaia import ARMP_G
from sklearn.base import clone

data = h5py.File("/Volumes/Transcend/data_sets/CN_isobutane_model/pruned_cn_isobutane/train_isopentane_cn_dft.hdf5", "r")

xyz = np.array(data.get("xyz"))
ene = np.array(data.get("ene"))*4.184
ene = ene - ene[0]
zs = np.array(data["zs"], dtype = int)
forces = np.array(data.get("forces"))*4.184

n_samples = 10

idx = list(range(n_samples))
splitter = modsel.KFold(n_splits=3, random_state=42, shuffle=True)
indices = splitter.split(idx)

acsf_params = {"radial_rs":np.arange(0,10, 0.6), "angular_rs":np.arange(0.5, 10.5, 0.6), "theta_s": np.arange(0, 3.14, 0.5), "eta":4.0, "zeta":8.0}
estimator = ARMP_G(iterations=2, representation_name='acsf', representation_params=acsf_params, batch_size=512,
                   tensorboard=True, tensorboard_subdir=os.getcwd(), hidden_layer_sizes=(50,30,10),
                   l1_reg=1.273530531541467e-10, l2_reg=0.06740360714223229, learning_rate=0.01, store_frequency=10)

# estimator.set_xyz(xyz[:n_samples])
# estimator.set_classes(zs[:n_samples])
# estimator.set_properties(ene[:n_samples])
# estimator.set_gradients(forces[:n_samples])

all_scores = []
counter = 0

for item in indices:

    dolly = clone(estimator)

    dolly.set_xyz(xyz[:n_samples])
    dolly.set_classes(zs[:n_samples])
    dolly.set_properties(ene[:n_samples])
    dolly.set_gradients(forces[:n_samples])

    dolly.generate_representation(method='fortran')

    idx_train = item[0]
    idx_test = item[1]

    print("Started training. \n")
    dolly.fit(idx_train)

    score_train = dolly.score(idx_train)
    score_test = dolly.score(idx_test)

    print("The mean absolute error for model %i on the train and test set is %s and %s (kJ/mol).\n" % (counter+1, str(-score_train), str(-score_test)))

    energies_predict, forces_predict = dolly.predict(idx_test)

    filename = "ef_pred_dev_" + str(counter+1) + ".npz"

    np.savez(filename, energies_predict, ene[idx_test],  forces_predict, forces[idx_test])

    counter +=1

    tf.reset_default_graph()