
from qml.aglaia.aglaia import ARMP
import glob
import numpy as np
import os
from sklearn import model_selection as modsel

## ------------- ** Loading the data ** ---------------

current_dir = os.path.dirname(os.path.realpath(__file__))
filenames = glob.glob("/Volumes/Transcend/repositories/my_qml_fork/qml/test/CN_isobutane/*.xyz")
energies = np.loadtxt("/Volumes/Transcend/repositories/my_qml_fork/qml/test/CN_isobutane/prop_kjmol_training.txt", usecols=[1])
filenames.sort()

## ------------- ** Setting up the estimator ** ---------------

acsf_params = {"nRs2": 5, "nRs3": 5, "nTs": 5, "rcut": 5, "acut": 5, "zeta": 220.127, "eta": 30.8065}
estimator = ARMP(iterations=5000, representation_name='acsf', representation_params=acsf_params, tensorboard=True,
                 learning_rate=0.075, l1_reg=0.0, l2_reg=0.0)

estimator.generate_compounds(filenames)
estimator.set_properties(energies)

estimator.generate_representation(method="fortran")
print("The shape of the representation is: %s" % (str(estimator.representation.shape)))

##  ------------- ** Fitting to the data ** ---------------

idx = np.arange(0,100)
idx_train, idx_test = modsel.train_test_split(idx, test_size=0, random_state=42, shuffle=True)

estimator.fit(idx_train)

##  ------------- ** Predicting and scoring ** ---------------

score = estimator.score(idx_train)

print("The mean absolute error is %s kJ/mol." % (str(-score)))

energies_predict = estimator.predict(idx_train)