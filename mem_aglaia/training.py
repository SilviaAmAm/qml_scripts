import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
from qml.aglaia.aglaia import ARMP_G
import h5py
from sklearn import model_selection as modsel

data = h5py.File("/Volumes/Transcend/data_sets/CN_isopentane/pruned_dft_with_forces/pruned_isopentane_cn_dft.hdf5", "r")

n_samples = 100

xyz = np.array(data.get("xyz")[:n_samples], dtype=np.float32)
ene = np.array(data.get("ene")[:n_samples], dtype=np.float32)*2625.50
ene = ene - data.get("ene")[0]*2625.50
zs = np.array(data["zs"][:n_samples], dtype=np.int32)
forces = np.array(data.get("forces")[:n_samples], dtype=np.float32)

acsf_params = {"nRs2":5, "nRs3":5, "nTs":5, "rcut":5, "acut":5, "zeta":220.127, "eta":30.8065}
estimator = ARMP_G(iterations=5000, l1_reg=0.0, l2_reg=0.0, learning_rate=0.075,
                   representation_name='acsf', representation_params=acsf_params, tensorboard=True, store_frequency=2,
                     method='fortran')

estimator.set_xyz(xyz)
estimator.set_classes(zs)
estimator.set_properties(ene)
estimator.set_gradients(forces)

idx = np.arange(0, n_samples)
idx_train, idx_test = modsel.train_test_split(idx, test_size=0, random_state=42, shuffle=True)

# estimator.load_nn()

estimator.fit(idx_train)
# print("Done the fitting")

ene_pred, f_pred = estimator.predict(idx_train)

plt.scatter(ene_pred, ene[idx_train])
plt.xlabel("Predicted energies (kJ/mol)")
plt.ylabel("DFT energies (kJ/mol)")
# plt.savefig("mem_aglaia_overfit.png", dpi=200)
plt.show()

score = estimator.score(idx_train)
# print("\n The score is %s" % (str(score)))

# os.remove("predict.tfrecords")
# os.remove("training.tfrecords")

# estimator.save_nn()