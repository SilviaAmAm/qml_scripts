import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['legend.handlelength'] = 0
import seaborn as sns
sns.set()

data_1 = np.genfromtxt('deep_batch500_prefetch3_shuffle0515.csv', delimiter=',', skip_header=1)
data_2 = np.genfromtxt('deep_batch750_prefetch2_shuffle0515.csv', delimiter=',', skip_header=1)
data_3 = np.genfromtxt('deep_batch500_prefetch3_shuffle2535.csv', delimiter=',', skip_header=1)
data_4 = np.genfromtxt('shallow_batch100_prefetch2_shuffle0_2.csv', delimiter=',', skip_header=1)

idx_gpu1 = np.ix_(range(1,data_1.shape[0],2))
idx_gpu0 = np.ix_(range(0,data_1.shape[0],2))

load_1_gpu1 = data_1[idx_gpu1, 1][0][:100]
load_2_gpu1 = data_2[idx_gpu1, 1][0][:100]
load_3_gpu1 = data_3[idx_gpu1, 1][0][:100]
load_4_gpu1 = data_4[idx_gpu0, 1][0][:100]
x = range(100)

f, axarr = plt.subplots(4, sharex=True, figsize=(12,6))
axarr[0].plot(x, load_1_gpu1, label="samples=4500\nbatch=500\nprefetch=3\nshuffle=1")
axarr[0].legend(bbox_to_anchor=(1, 1))
axarr[1].plot(x, load_2_gpu1, label="samples=4500\nbatch=750\nprefetch=2\nshuffle=1")
axarr[1].legend(bbox_to_anchor=(1, 1))
axarr[2].plot(x, load_3_gpu1, label="samples=4500\nbatch=500\nprefetch=3\nshuffle=3")
axarr[2].legend(bbox_to_anchor=(1, 1))
axarr[3].plot(x, load_4_gpu1, label="samples=1000\nbatch=100\nprefetch=2\nshuffle=0")
axarr[3].legend(bbox_to_anchor=(1, 1))
axarr[3].set_xlabel('Time (s)')
for i in range(4):
    axarr[i].set_ylabel('GPU load (%)')
# plt.savefig("aglaia_forces_gpu.png", dpi=200)
plt.show()