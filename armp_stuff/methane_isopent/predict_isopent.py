from qml.aglaia.aglaia import ARMP
from qml.qmlearn.preprocessing import AtomScaler
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn import model_selection as modsel


# Getting the dataset
data_isopent = h5py.File("/Volumes/Transcend/data_sets/CN_isopentane/pruned_dft_with_forces/pruned_isopentane_cn_dft.hdf5", "r")
data_methane = h5py.File("/Volumes/Transcend/data_sets/CN_methane/dft/methane_cn_dft.hdf5", "r")

n_samples = 50

xyz_isopent = np.array(data_isopent.get("xyz")[:n_samples])
ene_isopent = np.array(data_isopent.get("ene")[:n_samples]) * 2625.50
ref_ene = data_isopent.get("ene")[0] * 2625.50
ene_isopent = ene_isopent - ref_ene
zs_isopent = np.array(data_isopent.get("zs")[:n_samples], dtype=np.int32)

scaling = AtomScaler()
ene_isopent_scaled = scaling.fit_transform(zs_isopent, ene_isopent)

xyz_methane = np.array(data_methane.get("xyz")[:n_samples])
ene_methane = np.array(data_methane.get("ene")[:n_samples]) * 2625.50
ene_methane = ene_methane - ref_ene
zs_methane = np.array(data_methane.get("zs")[:n_samples], dtype=np.int32)

ene_methane_scaled = scaling.fit_transform(zs_methane, ene_methane)

pad_xyz_methane = np.concatenate((xyz_methane, np.zeros((xyz_methane.shape[0], xyz_isopent.shape[1] - xyz_methane.shape[1], 3))), axis=1)
pad_zs_methane = np.concatenate((zs_methane, np.zeros((zs_methane.shape[0], zs_isopent.shape[1] - zs_methane.shape[1]), dtype=np.int32)), axis=1)

concat_xyz = np.concatenate((xyz_isopent, pad_xyz_methane))
concat_ene = np.concatenate((ene_isopent_scaled, ene_methane_scaled))
concat_zs = np.concatenate((zs_isopent, pad_zs_methane))

print("Padded the data")

acsf_params={"nRs2":5, "nRs3":5, "nTs":5, "rcut":5, "acut":5, "zeta":220.127, "eta":30.8065}

# Generate estimator
estimator = ARMP(iterations=500, l1_reg=0.0, l2_reg=0.0, learning_rate=0.00005, representation_name='acsf',
                 representation_params=acsf_params, tensorboard=True, store_frequency=10, hidden_layer_sizes=(50,30,10), batch_size=20)

estimator.set_properties(concat_ene)
estimator.generate_representation(concat_xyz, concat_zs, method='fortran')

print("Generated the representations")
print(estimator.representation.shape)

idx = list(range(2*n_samples))
idx_train, idx_test = modsel.train_test_split(idx, random_state=42, shuffle=True)

estimator.fit(idx_train)

score = estimator.score(idx_train)
print(score)

pred = estimator.predict(idx_train)

plt.scatter(pred, concat_ene[idx_train])
plt.show()