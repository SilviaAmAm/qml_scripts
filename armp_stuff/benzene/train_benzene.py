import numpy as np
from qml.aglaia.aglaia import ARMP
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

data = np.load("/Volumes/Transcend/data_sets/benzene_dft.npz")
xyz = data["R"]
ene = data["E"].ravel() * 4.184
ref_ene = np.mean(ene)
ene = ene - ref_ene
zs = np.reshape(np.tile(data["z"], reps=xyz.shape[0]), newshape=(xyz.shape[0], xyz.shape[1]))

plt.scatter(list(range(len(ene))), ene, alpha=0.6)
plt.show()

#
# acsf_param = {"nRs2": 5, "nRs3": 5, "nTs": 5, "rcut": 5, "acut": 5, "zeta": 220.127, "eta": 30.8065}
# estimator = ARMP(iterations=1000, batch_size=512, l1_reg=0.0, l2_reg=0.0, learning_rate=0.001, representation_name='acsf',
#                  representation_params=acsf_param, tensorboard=False, store_frequency=50)
# estimator.set_properties(ene)
# estimator.generate_representation(xyz, zs, method='fortran')


