import os
import glob
import numpy as np
from qml.aglaia.aglaia import ARMP_G
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.realpath(__file__))
data = np.load(current_dir+"/../test/data/CN_isopentane_forces.npz")

xyz = data["arr_0"]
zs = data["arr_1"]
ene = data["arr_2"]
forces = data["arr_3"]

print(xyz.shape, zs.shape, ene.shape, forces.shape)

## ------------- ** Setting up the estimator ** ---------------

estimator = ARMP_G(iterations=1, representation='acsf', representation_params={"radial_rs": np.arange(0,10, 2), "angular_rs": np.arange(0.5, 10.5, 2),
"theta_s": np.arange(0, 3.14, 2.5)}, tensorboard=False, store_frequency=1, batch_size=10)

estimator.set_xyz(xyz[:2])
estimator.set_classes(zs[:2])
estimator.set_properties(ene[:2])
estimator.set_gradients(forces[:2])

estimator.generate_representation()
print(estimator.representation.shape[0])

idx = np.arange(0, 2)
estimator.fit(idx)

ene_1, f_1 = estimator.predict(idx)
ene_2, f_2 = estimator.predict_from_xyz(xyz[:2], zs[:2])

f_1_rp = np.reshape(f_1, (f_1.shape[0]*f_1.shape[1]*f_1.shape[2],))
f_2_rp = np.reshape(f_2, (f_1.shape[0]*f_1.shape[1]*f_1.shape[2],))

plt.scatter(f_1_rp, f_2_rp)
plt.show()