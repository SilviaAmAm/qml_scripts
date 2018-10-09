"""
This script plots the difference in time for an iteration of training for two different models
"""

import matplotlib.pyplot as plt
from numpy import genfromtxt
import numpy as np
import seaborn as sns
sns.set()

data_tfrecord = genfromtxt("/Volumes/Transcend/repositories/my_qml_fork/qml_scripts/Analysis/loss_tfrecord.csv", delimiter=',', skip_header=1)
data_no_tfrecord = genfromtxt("/Volumes/Transcend/repositories/my_qml_fork/qml_scripts/Analysis/loss_notfrecord.csv", delimiter=',', skip_header=1)
data_xyz_toenef = genfromtxt("/Volumes/Transcend/repositories/my_qml_fork/qml_scripts/Analysis/loss_xyztoenef.csv", delimiter=',', skip_header=1)

data_tfrecord[:, 0] = data_tfrecord[:, 0] - data_tfrecord[0, 0]
data_no_tfrecord[:, 0] = data_no_tfrecord[:, 0] - data_no_tfrecord[0, 0]
data_xyz_toenef[:, 0] = data_xyz_toenef[:, 0] - data_xyz_toenef[0, 0]

# time per 20 iteration
tpi_tfrecord = []
tpi_no_tfrecord = []
tpi_xyztoenef = []

for i in range(data_tfrecord.shape[0]-1):
    tpi_tfrecord.append(data_tfrecord[i + 1, 0] - data_tfrecord[i, 0])
    tpi_no_tfrecord.append(data_no_tfrecord[i+1, 0] - data_no_tfrecord[i, 0])

for i in range(data_xyz_toenef.shape[0]-1):
    tpi_xyztoenef.append(data_xyz_toenef[i + 1, 0] - data_xyz_toenef[i, 0])

plt.scatter(range(len(tpi_tfrecord)), tpi_tfrecord, label="With tfrecord")
plt.scatter(range(len(tpi_no_tfrecord)), tpi_no_tfrecord, label="Without tfrecord")
plt.scatter(range(len(tpi_xyztoenef)), tpi_xyztoenef, label="One go")
plt.xlabel("Iteration number / 20")
plt.ylabel("Duration (s)")
plt.legend()
plt.savefig("iteration_time.png", dpi=200)
plt.show()

mean_tpi_tfrecord = np.mean(tpi_tfrecord)
err_tpi_tfrecord = np.std(tpi_tfrecord)

mean_tpi_no_tfrecord = np.mean(tpi_no_tfrecord)
err_tpi_no_tfrecord = np.std(tpi_no_tfrecord)

mean_tpi_xyztoenef = np.mean(tpi_xyztoenef)
err_tpi_xyztoenef = np.std(tpi_xyztoenef)

means = [mean_tpi_tfrecord, mean_tpi_no_tfrecord, mean_tpi_xyztoenef]
x = range(len(means))
errs = [err_tpi_tfrecord, err_tpi_no_tfrecord, err_tpi_xyztoenef]
xticks = ["TFR", "No TFR", "One go"]

p2 = plt.bar(x, means, 0.4, yerr=errs)
plt.xticks(x, xticks)
plt.ylabel('20 iterations time (s)')

plt.show()

xticks = ["TFR", "No TFR", "One go"]
full_training_time = [64.0831, 57.2011, 34.4975 ]

p3 = plt.bar(x, full_training_time, 0.4)
plt.xticks(x, xticks)
plt.ylabel('Full training time (min)')
plt.savefig("training_time.png", dpi=200)
plt.show()

