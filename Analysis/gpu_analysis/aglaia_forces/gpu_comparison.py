"""
This script compares the different usage of the two GPUs on KRatos
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['legend.handlelength'] = 0
import seaborn as sns
sns.set()

data_0 = np.genfromtxt('shallow_batch100_prefetch2_shuffle0_2.csv', delimiter=',', skip_header=1)
data_1 = np.genfromtxt('shallow_batch100_prefetch2_shuffle0.csv', delimiter=',', skip_header=1)

idx_gpu1 = np.ix_(range(1,data_1.shape[0],2))
idx_gpu0 = np.ix_(range(0,data_0.shape[0],2))

load_0 = data_0[idx_gpu0, 1][0][:100]
load_1 = data_1[idx_gpu1, 1][0][:100]
x = range(100)

f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(x, load_0)
axarr[0].set_title("GPU0")
axarr[0].set_ylim(-5,105)
axarr[1].plot(x, load_1)
axarr[1].set_title("GPU1")
axarr[1].set_xlabel('Time (s)')
axarr[1].set_ylim(-5,105)
for i in range(2):
    axarr[i].set_ylabel('GPU load (%)')
plt.savefig("comparing_gpus.png", dpi=200)
plt.show()
