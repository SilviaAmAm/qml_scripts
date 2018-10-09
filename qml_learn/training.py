from qml import qmlearn
import numpy as np
import sklearn.pipeline
np.random.seed(0)

data = qmlearn.Data("/Volumes/Transcend/repositories/my_qml_fork/qml/test/qm7/*.xyz")
energies = np.loadtxt("/Volumes/Transcend/repositories/my_qml_fork/qml/test/data/hof_qm7.txt", usecols=1)
data.set_energies(energies)

# Create model
model = sklearn.pipeline.make_pipeline(
    qmlearn.preprocessing.AtomScaler(data),
    qmlearn.representations.AtomCenteredSymmetryFunctions(),
    qmlearn.models.NeuralNetwork(iterations=9000, batch_size=50, learning_rate=0.05, hl1=50, hl2=30, hl3=10),
)

indices = np.arange(1000)
np.random.shuffle(indices)

model.fit(indices[:100])

scores = model.score(indices[:100])
print("Negative MAE:", scores)
