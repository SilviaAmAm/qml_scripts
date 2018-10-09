from qml.aglaia.aglaia import ARMP
import os
import numpy as np

## ------------- ** Loading the data ** ---------------

current_dir = os.path.dirname(os.path.realpath(__file__))
data = np.load("/Volumes/Transcend/repositories/my_qml_fork/qml/test/data/local_slatm_ch4cn_light.npz")

representation = data["arr_0"]
zs = data["arr_1"]
energies = data["arr_2"]

## ------------- ** Setting up the estimator ** ---------------

estimator = ARMP(iterations=3000, learning_rate=0.075, l1_reg=0.0, l2_reg=0.0, tensorboard=True, store_frequency=50)

##  ------------- ** Fitting to the data ** ---------------

estimator.fit(x=representation, y=energies, classes=zs)

##  ------------- ** Predicting and scoring ** ---------------

score = estimator.score(x=representation, y=energies, classes=zs)

print("The mean absolute error is %s kJ/mol." % (str(-score)))

energies_predict = estimator.predict(x=representation, classes=zs)
