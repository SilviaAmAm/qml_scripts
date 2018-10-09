from qml.aglaia.aglaia import ARMP_G
import numpy as np
import h5py
import matplotlib.pyplot as plt

f = open("/Volumes/Transcend/data_sets/relaxed_scan_24/00.xyz", "r")
counter = 0
dic = {'H':1, 'C':6, 'N':7}

xyz = []
zs = []

for line in f:
    if counter <= 1:
        counter += 1
        continue
    split_line = line.split()
    xyz_atom = [float(split_line[1]), float(split_line[2]), float(split_line[3])]
    zs.append(dic[split_line[0]])
    xyz.append(xyz_atom)

    counter += 1

for i in range(19):
    xyz.append([0, 0, 0])
    zs.append(0)

estimator = ARMP_G(iterations=50, representation='acsf', representation_params={"nRs2": 5, "nRs3": 5,
"nTs": 2}, tensorboard=False, store_frequency=1, batch_size=20)

estimator.load_nn("saved_model")

xyz = np.asarray(xyz)
xyz = np.reshape(xyz, (1, xyz.shape[0], xyz.shape[1]))
zs = np.asarray(zs, dtype=np.int32)
zs = np.reshape(zs, (1, zs.shape[0]))
print(xyz.shape, zs.shape)

ene_2, f_2 = estimator.predict_from_xyz(xyz, zs)



