from qml.aglaia.aglaia import ARMP_G
import h5py
import numpy as np
import matplotlib.pyplot as plt

data = h5py.File("/Volumes/Transcend/data_sets/CN_isopentane/pruned_dft_with_forces/pruned_isopentane_cn_dft.hdf5", "r")

n_samples = 10

xyz = np.array(data.get("xyz")[:n_samples])
ene = np.array(data.get("ene")[:n_samples])*2625.50
ene = ene - data.get("ene")[0]*2625.50
zs = np.array(data["zs"][:n_samples], dtype=np.int32)
forces = np.array(data.get("forces")[:n_samples])

# Creating the estimator
estimator = ARMP_G(iterations=10, batch_size=30, hidden_layer_sizes=(40, 20, 10), l2_reg=0.0, tensorboard=False,
                   store_frequency=5, learning_rate=0.001, representation_params={'nRs2':7, 'nRs3':7, 'nTs':7,
                                                                                    'zeta':220.127, 'eta':30.8065})

estimator.set_xyz(xyz)
estimator.set_properties(ene)
estimator.set_classes(zs)
estimator.set_gradients(forces)

idx = list(range(n_samples))
estimator.load_nn("saved_model")
estimator.fit(idx)

pred_ene, pred_forces = estimator.predict(idx)

score = estimator.score(idx)

# estimator.save_nn()

plt.scatter(ene, pred_ene)
plt.show()