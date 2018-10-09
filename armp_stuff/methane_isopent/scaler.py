import h5py
import numpy as np
from qml.qmlearn.preprocessing import AtomScaler
import matplotlib.pyplot as plt

data_isopent = h5py.File("/Volumes/Transcend/data_sets/CN_isopentane/pruned_dft_with_forces/pruned_isopentane_cn_dft.hdf5", "r")
data_methane = h5py.File("/Volumes/Transcend/data_sets/CN_methane/dft/methane_cn_dft.hdf5", "r")
data_squal = h5py.File("/Volumes/Transcend/data_sets/CN_squalane/dft/squalane_cn_dft.hdf5", "r")

n_samples = 15000

xyz_isopent = np.array(data_isopent.get("xyz")[:n_samples])
ene_isopent = np.array(data_isopent.get("ene")[:n_samples])* 2625.50
ref_ene = data_isopent.get("ene")[0] * 2625.50
ene_isopent = ene_isopent - ref_ene
zs_isopent = np.array(data_isopent.get("zs")[:n_samples], dtype=np.int32)

xyz_methane = np.array(data_methane.get("xyz")[:n_samples])
ene_methane = np.array(data_methane.get("ene")[:n_samples])* 2625.50
ene_methane = ene_methane - ref_ene
zs_methane = np.array(data_methane.get("zs")[:n_samples], dtype=np.int32)

xyz_squal = np.array(data_squal.get("xyz"))
zs_squal = np.array(data_squal.get("zs"))
ene_squal = np.array(data_squal.get("ene")) * 2625.50
ene_squal = ene_squal - ref_ene


concat_ene = np.concatenate((ene_methane, ene_isopent, ene_squal))
concat_zs =   list(zs_methane) + list(zs_isopent) + list(zs_squal)

scaling = AtomScaler()
concat_ene_scaled = scaling.fit_transform(concat_zs, concat_ene)

plt.scatter(range(len(concat_ene_scaled)), concat_ene_scaled)
plt.show()