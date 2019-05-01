import struct
import numpy as np
print("asdfw")
# import the data
source_file = '/fasterHome/workingDataDir/shattering/shatteredMNISTsoftplus2_4shearLevels.bin'
label_file = '/fasterHome/workingDataDir/shattering/MNIST_labels_juliaOrdered.bin'

# load the labels
with open(label_file, mode='rb') as file:
    file_content = file.read()
nEx = struct.unpack('q', file_content[:8])[0]
labels = struct.unpack(str(nEx) + 'q', file_content[8:])

# load the data
with open(source_file, mode='rb') as file:
    file_content = file.read()
dim,nEx = struct.unpack('qq', file_content[:16])
file_content = 3                # garbage collection, otherwise this eats a lot of space
tmp = np.fromfile(source_file, np.float32, -1)
tmp = tmp[4:]
stCoeff = tmp.reshape(dim, nEx)

from sklearn import svm
classifier = svm.SVC(kernel='poly', degree=3)
from sklearn.model_selection import train_test_split
stCoeff = stCoeff.reshape(70000, -1)
dTrain, dTest, lTrain, lTest = train_test_split(stCoeff, labels, random_state=4025)
classifier.fit(dTrain, lTrain)
classifier.score(dTest, lTest)

import Pickle



import matplotlib.pyplot as plt
plt.plot(classifier.support_vectors_)
