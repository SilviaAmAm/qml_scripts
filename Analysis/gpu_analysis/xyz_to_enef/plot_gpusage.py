import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import matplotlib
matplotlib.rcParams['legend.handlelength'] = 0

data_1 = np.genfromtxt('gpu_log.csv', delimiter=',', skip_header=1)

idx_gpu1 = np.ix_(range(1,data_1.shape[0],2))

load_1_gpu1 = data_1[idx_gpu1, 1][0]

x = range(len(load_1_gpu1))

f, axarr = plt.subplots(1, figsize=(9,5))
axarr.plot(x, load_1_gpu1, label="batch=100\nprefetch=2\nshuffle=4")
axarr.legend(bbox_to_anchor=(0.17, 0.2))
axarr.set_xlabel('Time (s)')
axarr.set_ylabel('GPU load (%)')
plt.savefig("onego_gpu.png", dpi=200)
plt.show()