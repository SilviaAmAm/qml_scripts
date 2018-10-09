import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

data = np.load("/Volumes/Transcend/calculations/cn_isopentane/result_analysis/pred_squal.npz")
ene_nn = data["arr_0"]
ene_dft = data["arr_1"] + 2577900

MAE = np.sum(np.abs(ene_nn - ene_dft))/len(ene_nn)
print(MAE)


plt.figure(figsize=(7,5))
plt.scatter(ene_dft, ene_nn)
# plt.xlim(-1800-2577900, 850-2577900)
# plt.ylim(-1800, 850)
plt.xlim(-1800, 850)
plt.ylim(-1800, 850)
plt.ylabel("Predicted Energies (kJ/mol)")
plt.xlabel("DFT Energies (kJ/mol)")
plt.savefig("squalane_lessbad.png", dpi=200)
plt.show()