import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

data = np.load("learning_curve.npz")
mae = data["arr_0"] * (-1)
samples = data["arr_1"]
print(mae)
print(samples)

plt.plot(samples, mae)
plt.xlabel("Training set size")
plt.ylabel("MAE (kJ/mol)")
# plt.show()
# plt.savefig("learning_curve.png", dpi=200)
