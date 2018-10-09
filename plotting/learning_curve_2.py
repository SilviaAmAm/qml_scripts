import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


filenames = []
for i in range(1,8):
    filenames.append("loss_" + str(i) + ".csv")

all_data = []

data = np.load("learning_curve.npz")
mae = data["arr_0"] * (-1)
samples = data["arr_1"]

for file in filenames:
    data = np.loadtxt(file, skiprows=1, delimiter=",")
    all_data.append(data[:, 1:])

for i, run in enumerate(all_data):
    lab = str(samples[i]) + " samples"
    plt.plot(run[:, 0], run[:,1], label=lab)

plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Cost function")
plt.yscale("log")
# plt.show()

plt.savefig("cost_lf.png", dpi=200)

