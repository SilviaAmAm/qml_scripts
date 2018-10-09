import pickle
import numpy as np
import h5py
from qml.aglaia.aglaia import ARMP
from sklearn import model_selection as modsel
import tensorflow as tf

data = h5py.File("/Volumes/Transcend/data_sets/CN_isobutane_model/pruned_dft_with_forces/pruned_isopentane_cn_dft.hdf5", "r")

n_samples = 5000

xyz = np.array(data.get("xyz")[:n_samples])
ene = np.array(data.get("ene")[:n_samples])*2625.5
ene = ene - ene[0]
zs = np.array(data["zs"][:n_samples], dtype = int)

learning_rate =  np.power(10, (-1 * np.random.uniform(low=2, high=5, size=(3,))))
l1_reg = np.power(10, (-1 * np.random.uniform(low=1, high=4, size=(3,))))
l2_reg = np.power(10, (-1 * np.random.uniform(low=1, high=4, size=(3,))))

print("The learning rate values:")
print(learning_rate)
print("The l1 regularisation values:")
print(l1_reg)
print("The l2 regularisation values:")
print(l2_reg)

acsf_params={"radial_rs": np.arange(0, 10, 0.5),"angular_rs": np.arange(0, 10, 0.5), "theta_s": np.arange(0, 3.14, 0.25)}

estimator = ARMP(iterations=2000, batch_size=256, l1_reg=0.0001, l2_reg=0.005, learning_rate=0.00015, representation='acsf',
                 representation_params=acsf_params, tensorboard=True, store_frequency=50)

estimator.set_properties(ene)
estimator.generate_representation(xyz, zs)

idx = list(range(n_samples))
idx_train, idx_test = modsel.train_test_split(idx, test_size=0.15, random_state=42, shuffle=True)

all_scores = []

for lr in learning_rate:
    for l1  in l1_reg:
        for l2 in l2_reg:

            estimator.fit(idx_train)
            score = estimator.score(idx_test)
            print("\n The model trained with learning rate %s, l1 reg %s and l2 reg %s has a test set MAE of %s kJ/mol. \n"
                  % (str(lr), str(l1), str(l2), str(score)))

            tf.reset_default_graph()





