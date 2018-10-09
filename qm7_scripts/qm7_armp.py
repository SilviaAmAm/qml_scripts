from qml.aglaia.aglaia import ARMP
import glob
import numpy as np
from sklearn import model_selection as modsel

test_dir = "/Volumes/Transcend/repositories/my_qml_fork/qml/test/"


filenames = glob.glob(test_dir + "/qm7/*.xyz")
energies = np.loadtxt(test_dir + '/data/hof_qm7.txt',
                      usecols=[1])
filenames.sort()

n_samples=500

estimator = ARMP(representation_name="acsf", iterations=100)
estimator.generate_compounds(filenames[:n_samples])
estimator.set_properties(energies[:n_samples])
estimator.generate_representation(method="fortran")

idx = np.arange(0, n_samples)
idx_train, idx_test = modsel.train_test_split(idx, random_state=42, shuffle=True, test_size=0.1)

estimator.fit(idx_train)

estimator.score(idx_train)


