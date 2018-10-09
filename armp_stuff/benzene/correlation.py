import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

data = np.load("pred_benzene.npz")
ene_nn = data["arr_0"]
ene_dft = data["arr_1"]

plt.figure(figsize=(7,5))
plt.scatter(ene_dft, ene_nn)
plt.title("Benzene predictions")
plt.xlim(-60, 100)
plt.ylim(-60, 100)
plt.ylabel("Predicted Energies (kJ/mol)")
plt.xlabel("DFT Energies (kJ/mol)")
# plt.savefig("squalane.png", dpi=200)
plt.show()