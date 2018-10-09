"""
This example shows how to use ARMP to overfit 100 data-points for the QM7 data set. It uses the Atom Centred Symmetry
functions as the representation.

This example takes about 3.5 min to run.
"""

from qml.aglaia.aglaia import ARMP
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection as modsel

filenames = sorted(glob.glob("/Volumes/Transcend/repositories/my_qml_fork/qml/test/qm7/*.xyz"))
energies = np.loadtxt("/Volumes/Transcend/repositories/my_qml_fork/qml/test/data/hof_qm7.txt", usecols=[1])
n_samples = len(filenames)
print("%i files were loaded." % (n_samples))

acsf_params = {"nRs2": 5, "nRs3": 5, "nTs": 5, "rcut": 5, "acut": 5, "zeta": 220.127, "eta": 30.8065}
estimator = ARMP(iterations=6000, representation_name='acsf', representation_params=acsf_params, l1_reg=0.0, l2_reg=0.0,
                 scoring_function="rmse", tensorboard=False, store_frequency=10, learning_rate=0.075)

estimator.set_properties(energies[:100])
estimator.generate_compounds(filenames[:100])
estimator.generate_representation(method="tf")
print(estimator.representation.shape)

idx = list(range(100))

idx_train, idx_test = modsel.train_test_split(idx, test_size=0, random_state=42, shuffle=True)

estimator.fit(idx_train)

score = estimator.score(idx_train)
print("The RMSE is %s kcal/mol." % (str(score)))

ene_pred = estimator.predict(idx_train)

plt.scatter(energies[idx_train], ene_pred)
plt.show()