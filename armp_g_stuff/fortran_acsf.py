import numpy as np
import h5py
from qml.aglaia.aglaia import ARMP_G
import matplotlib.pyplot as plt
from sklearn import model_selection as modsel

data = h5py.File("/Volumes/Transcend/data_sets/CN_isobutane_model/pruned_dft_with_forces/pruned_isopentane_cn_dft.hdf5", "r")

n_samples = 1000

xyz = np.array(data.get("xyz")[:n_samples])
ene = np.array(data.get("ene")[:n_samples])*2625.5
ene = ene - ene[0]
zs = np.array(data["zs"][:n_samples], dtype = int)
forces = np.array(data.get("forces")[:n_samples])*2625.5

acsf_param = {"nRs2": 5, "nRs3": 5, "nTs": 5, "rcut": 5, "acut": 5, "zeta": 220.127, "eta": 30.8065}
estimator = ARMP_G(iterations=100, representation_name='acsf', representation_params=acsf_param, tensorboard=True,
                   hidden_layer_sizes=(50,30,10), l1_reg=0, l2_reg=0, store_frequency=20, learning_rate=0.005)

# estimator.load_nn()

estimator.set_xyz(xyz)
estimator.set_classes(zs)
# estimator.generate_representation(xyz, zs, method='fortran')
estimator.set_properties(ene)
estimator.set_gradients(forces)

estimator.generate_representation(method='fortran')
print("Generated all representations.")

idx = list(range(n_samples))

idx_train, idx_test = modsel.train_test_split(idx, test_size=0.2, random_state=42, shuffle=True)

estimator.fit(idx_train)
# print("Fitted the model.")

predictions_ene, predictions_forces = estimator.predict(idx_test)
# predictions_ene_2, predictions_forces_2 =  estimator.predict_from_xyz(xyz, zs)

# assert np.all(np.isclose(predictions_ene, predictions_ene_2, rtol=1.e-6))

score = estimator.score(idx_test)
print(score)


plt.scatter(predictions_ene, ene[idx_test])
plt.show()

#
# plt.scatter(predictions_ene_2, ene)
# plt.show()
#
estimator.save_nn()

# plt.scatter(predictions_ene, predictions_ene_2)
# plt.show()
