from qml.aglaia.aglaia import ARMP_G
import numpy as np

acsf_params = {"nRs2":10, "nRs3":10, "nTs": 5, "eta2":4.0, "eta3":4.0, "zeta":8.0}
estimator = ARMP_G(iterations=5, representation='acsf', representation_params=acsf_params, batch_size=5, l1_reg=0.1, l2_reg=0.3)

parameters = estimator.get_params()

print(parameters["l1_reg"], parameters["l2_reg"])

print(parameters["representation_params"])