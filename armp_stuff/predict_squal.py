from qml.aglaia.aglaia import ARMP
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn import model_selection as modsel
import tensorflow as tf

# Getting the dataset
data_isopent = h5py.File("/Volumes/Transcend/data_sets/CN_isopentane/pruned_dft_with_forces/pruned_isopentane_cn_dft.hdf5", "r")

n_samples = 10

xyz_isopent = np.array(data_isopent.get("xyz")[-n_samples:])
ene_isopent = np.array(data_isopent.get("ene")[-n_samples:]) * 2625.50
ref_ene = data_isopent.get("ene")[0] * 2625.50
ene_isopent = ene_isopent - ref_ene
zs_isopent = np.array(data_isopent["zs"][-n_samples:], dtype=np.int32)

pad_xyz = np.concatenate((xyz_isopent, np.zeros((xyz_isopent.shape[0], 94 - xyz_isopent.shape[1], 3))), axis=1)
pad_zs = np.concatenate((zs_isopent, np.zeros((zs_isopent.shape[0], 94 - zs_isopent.shape[1]), dtype=np.int32)), axis=1)

print("Padded the data")

acsf_params={"nRs2":15, "nRs3":15, "nTs":15, "rcut":5, "acut":5, "zeta":220.127, "eta":30.8065}

# Generate estimator
estimator = ARMP(iterations=10, l1_reg=0.0001, l2_reg=0.005, learning_rate=0.0005, representation_name='acsf',
                 representation_params=acsf_params, tensorboard=True, store_frequency=2, hidden_layer_sizes=(50,30,10), batch_size=200)

estimator.set_properties(ene_isopent)
estimator.generate_representation(pad_xyz, pad_zs, method='fortran')

print("Generated the representations")
print(estimator.representation.shape)

idx = list(range(n_samples))
idx_train, idx_test = modsel.train_test_split(idx, random_state=42, shuffle=True)

estimator.fit(idx_train)

data_squal = h5py.File("/Volumes/Transcend/data_sets/CN_squalane/dft/squalane_cn_dft.hdf5", "r")

xyz_squal = np.array(data_squal.get("xyz")[:10])
zs_squal = np.array(data_squal.get("zs")[:10], dtype=np.int32)
ene_squal = np.array(data_squal.get("ene")[:10]) * 2625.50
ene_squal = ene_squal - ref_ene

estimator.score(idx_test)

pred1 = estimator.predict_from_xyz(xyz_squal, zs_squal)
print("Done squal pred")
pred2 = estimator.predict(idx_test)

# estimator.save_nn()
#
# print(pred1)
# print(pred2)
#
plt.scatter(pred2, pred1)
plt.show()