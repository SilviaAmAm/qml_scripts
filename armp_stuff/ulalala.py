import numpy as np
import shutil
from qml.aglaia.aglaia import ARMP
import tensorflow as tf

xyz = np.array([[[0, 1, 0], [0, 1, 1], [1, 0, 1]],
                    [[1, 2, 2], [3, 1, 2], [1, 3, 4]],
                    [[4, 1, 2], [0.5, 5, 6], [-1, 2, 3]]])
zs = np.array([[1, 2, 3],
               [1, 2, 3],
               [1, 2, 3]])

ene_true = np.array([0.5, 0.9, 1.0])

estimator = ARMP(iterations=10, l1_reg=0.0001, l2_reg=0.005, learning_rate=0.0005, representation='acsf',
                 representation_params={"radial_rs": np.arange(0, 10, 5), "angular_rs": np.arange(0, 10, 5),
                                        "theta_s": np.arange(0, 3.14, 3)},
                 tensorboard=True, store_frequency=10
                 )

estimator.set_properties(ene_true)
estimator.generate_representation(xyz, zs)

idx = list(range(xyz.shape[0]))

estimator.fit(idx)
estimator.save_nn(save_dir="temp")

pred1 = estimator.predict(idx)

estimator.loaded_model = True

estimator.fit(idx)

pred2 = estimator.predict(idx)
estimator.session.close()
tf.reset_default_graph()

new_estimator = ARMP(iterations=10, l1_reg=0.0001, l2_reg=0.005, learning_rate=0.0005, representation='acsf',
                 representation_params={"radial_rs": np.arange(0, 10, 5), "angular_rs": np.arange(0, 10, 5),
                                        "theta_s": np.arange(0, 3.14, 3)},
                    tensorboard=True, store_frequency=10
                     )
new_estimator.set_properties(ene_true)
new_estimator.generate_representation(xyz, zs)

new_estimator.load_nn("temp")

pred3 = new_estimator.predict(idx)

new_estimator.fit(idx)

pred4 = new_estimator.predict(idx)

shutil.rmtree("temp")
# shutil.rmtree("tb")
# shutil.rmtree("tensorboard")

assert np.all(pred1 == pred3)
assert np.all(pred2 == pred4)

