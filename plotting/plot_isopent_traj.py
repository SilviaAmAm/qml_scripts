import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()

data = np.load("pred_ene_4000_traj.npz", 'r')
ene_nn = data["arr_0"]
ene_dft = data["arr_1"]

line_1 = plt.scatter(list(range(len(ene_nn))), ene_nn, c=sns.xkcd_rgb["medium green"], label="Predictions", marker="o")
line_2 = plt.scatter(list(range(len(ene_nn))), ene_dft, c=sns.xkcd_rgb["denim blue"], label="True values", marker="o")
plt.xlabel("Time step")
plt.ylabel("Energy (kJ/mol)")
plt.legend(handles=[line_1, line_2])
plt.savefig("Lars_traj.png", dpi=200)
plt.show()