
"""
This script shows how to set up the ARMP estimator where the data to be fitted is passed directly to the fit function.
"""

import sys
from copy import deepcopy

import os
import numpy as np

import qml
from qml.aglaia.aglaia import ARMP


def get_energies(filename):
    """ Returns a dictionary with heats of formation for each xyz-file.
    """

    f = open(filename, "r")
    lines = f.readlines()
    f.close()

    energies = dict()

    for line in lines:
        tokens = line.split()

        xyz_name = tokens[0]
        hof = float(tokens[1])

        energies[xyz_name] = hof

    return energies


## ------------- ** Loading the data ** ---------------

if __name__ == "__main__":

    test_dir = os.path.dirname(os.path.realpath(__file__))

    # Parse file containing PBE0/def2-TZVP heats of formation and xyz filenames
    data = get_energies("/Volumes/Transcend/repositories/my_qml_fork/qml/test/data/hof_qm7.txt")

    # Generate a list of qml.data.Compound() objects"
    mols = []

    pad_template = np.zeros((23, 276))

    for xyz_file in sorted(data.keys()):
        # Initialize the qml.data.Compound() objects
        mol = qml.data.Compound("/Volumes/Transcend/repositories/my_qml_fork/qml/test/qm7/" + xyz_file)

        # Associate a property (heat of formation) with the object
        mol.properties = data[xyz_file]

        # This is a Molecular Coulomb matrix sorted by row norm
        mol.generate_atomic_coulomb_matrix(size=23, sorting="row-norm")
        # mol.generate_bob()

        rep = deepcopy(pad_template)
        rep[:mol.natoms] = mol.representation

        mol.representation = deepcopy(rep)
        z = np.zeros((23), dtype=np.int32)
        z[:mol.natoms] = mol.nuclear_charges
        mol.nuclear_charges = deepcopy(z)
        # print (mol.representation)
        mols.append(mol)

    # Shuffle molecules
    np.random.seed(666)
    np.random.shuffle(mols)

    # Make training and test sets
    n_train = 400
    n_test = 100

    training = mols[:n_train]
    test = mols[-n_test:]

    X = np.array([mol.representation for mol in training])
    Xs = np.array([mol.representation for mol in test])

    Z = np.array([mol.nuclear_charges for mol in training], dtype=np.int32)
    Zs = np.array([mol.nuclear_charges for mol in test])

    # List of properties
    Y = np.array([mol.properties for mol in training])
    Ys = np.array([mol.properties for mol in test])

    ## ------------- ** Setting up the estimator ** ---------------

    print(Z)
    print(Z.shape)

    estimator = ARMP(iterations=10,
                     l1_reg=0.0,
                     l2_reg=0.0,
                     hidden_layer_sizes=(40, 20, 10),
                     tensorboard=True,
                     store_frequency=10,
                     # batch_size=400,
                     batch_size=n_train,
                     learning_rate=0.1,
                     # scoring_function="mae",
                     )

    estimator.set_representations(representations=X)
    estimator.set_classes(Z)
    estimator.set_properties(Y)

    # idx = np.arange(0,100)

    # estimator.fit(idx)

    # score = estimator.score(idx)

    # estimator.fit(x=representation, y=energies, classes=zs)
    estimator.fit(x=X, y=Y, classes=Z)

    ##  ------------- ** Predicting and scoring ** ---------------

    score = estimator.score(x=X, y=Y, classes=Z)

    print("The mean absolute error is %s kJ/mol." % (str(-score)))

    # energies_predict = estimator.predict(idx)
    # print(energies_predict)