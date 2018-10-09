import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import matplotlib
matplotlib.rcParams['legend.handlelength'] = 0

data_1 = np.genfromtxt('deep_batch500_prefetch3_shuffle0515.csv', delimiter=',', skip_header=1)
data_2 = np.genfromtxt('deep_batch500_prefetch3_shuffle2535.csv', delimiter=',', skip_header=1)
data_3 = np.genfromtxt('deep_batch750_prefetch3_shuffle0515.csv', delimiter=',', skip_header=1)
data_4 = np.genfromtxt('deep_batch750_prefetch4_shuffle0515.csv', delimiter=',', skip_header=1)

idx_gpu1 = np.ix_(range(1,data_1.shape[0],2))

load_1_gpu1 = data_1[idx_gpu1, 1][0]
load_2_gpu1 = data_2[idx_gpu1, 1][0]
load_3_gpu1 = data_3[idx_gpu1, 1][0]
load_4_gpu1 = data_4[idx_gpu1, 1][0]
x = range(len(load_3_gpu1))

f, axarr = plt.subplots(4, sharex=True, figsize=(12,6))
axarr[0].plot(x, load_1_gpu1, label="batch=500\nprefetch=3\nshuffle=1")
axarr[0].legend(bbox_to_anchor=(1, 1))
axarr[1].plot(x, load_2_gpu1, label="batch=500\nprefetch=3\nshuffle=3")
axarr[1].legend(bbox_to_anchor=(1, 1))
axarr[2].plot(x, load_3_gpu1, label="batch=750\nprefetch=3\nshuffle=1")
axarr[2].legend(bbox_to_anchor=(1, 1))
axarr[3].plot(x, load_4_gpu1, label="batch=750\nprefetch=4\nshuffle=1")
axarr[3].legend(bbox_to_anchor=(1, 1))

axarr[3].set_xlabel('Time (s)')
for i in range(4):
    axarr[i].set_ylabel('GPU load (%)')
plt.savefig("mem_aglaia_gpu.png", dpi=200)
plt.show()