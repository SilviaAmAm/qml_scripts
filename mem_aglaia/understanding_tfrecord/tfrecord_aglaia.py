import os
import numpy as np
from qml.aglaia.aglaia import ARMP_G


current_dir = os.path.dirname(os.path.realpath(__file__))
data = np.load(current_dir+"/../../qml/test/data/CN_isopentane_forces.npz")

xyz = data["arr_0"]
zs = data["arr_1"]
ene = data["arr_2"]
forces = data["arr_3"]

## ------------- ** Setting up the estimator ** ---------------

estimator = ARMP_G(iterations=5, representation_name='acsf', representation_params={"nRs2": 5, "nRs3": 5,
"nTs": 2}, tensorboard=False, store_frequency=1, batch_size=20, method='fortran')

n_samples = 10

estimator.set_xyz(xyz[:n_samples])
estimator.set_classes(zs[:n_samples])
estimator.set_properties(ene[:n_samples])
estimator.set_gradients(forces[:n_samples])

idx = np.arange(0, n_samples)

estimator.fit(idx)
print("\n Done the fitting")

ene_1, f_1 = estimator.predict(idx)
print("\n Done the predictions")

score = estimator.score(idx)
print("\n The score is %s" % (str(score)))

ene_2, f_2 = estimator.predict_from_xyz(xyz[idx], zs[idx])

print(ene_1)
print(ene_2)