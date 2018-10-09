import sys
sys.path.insert(0, '/Volumes/Transcend/repositories/my_qml_fork/qml/qml/aglaia')
import symm_funct
import tf_utils
import numpy as np
import tensorflow as tf

xyz = np.array([[[0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]],
       [[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [-1, 2, 3]]])

zs = np.array([[0, 1, 2, 0], [0, 2, 0, 1]])

elements = np.array([0, 1, 2])
element_pairs = np.array([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]])

xyz_tf = tf.placeholder(shape=[4, 3], dtype=tf.float32)
zs_tf = tf.placeholder(shape=[4], dtype=tf.int32)

descriptor = symm_funct.generate_parkhill_acsf_single(xyz_tf, zs_tf, elements, element_pairs, rcut=4.0, acut=4.0, nRs2=3,
                                                      nRs3=3, nTs=2, zeta=1.0, eta2=1.0, eta3=1.0)

jacobian = tf_utils.partial_derivatives(descriptor, xyz_tf)

print(descriptor, jacobian)

per_atom_descriptor = tf.unstack(descriptor, axis=0)
per_atom_zs = tf.unstack(zs, axis=0)
per_atom_grad = tf.unstack(jacobian, axis=0)

weights = tf.Variable(tf.truncated_normal([45,1]))
bias = tf.Variable(tf.zeros([1]))

# model
atomic_energies = []
for i in range(len(per_atom_descriptor)):
    atomic_energy = tf.add(tf.matmul(descriptor, weights), bias)
    atomic_energies.append(atomic_energy)





