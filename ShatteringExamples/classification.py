import numpy as np
import h5py

# import the data
source_file = "/fasterHome/workingDataDir/shattering/shatteredMNISTabs2.h5"

# load the data
f = h5py.File(source_file, 'r')
db = f["data"]
labels = db["labels"][:]
stCoeff = db["shattered"]
stCoeff = stCoeff[:,:].transpose()

from sklearn import svm
classifier = svm.SVC(kernel='rbf', degree=3, verbose=True)
from sklearn.model_selection import train_test_split
dTrain, dTest, lTrain, lTest = train_test_split(stCoeff, labels, random_state=4025)
# for now, let's see how a model with just the first 100 entries does

nEntries = 100
classifier.fit(dTrain[:nEntries], lTrain[:nEntries])
classScore = classifier.score(dTest, lTest)

import Pickle



import matplotlib.pyplot as plt
plt.plot(classifier.support_vectors_)
plt.show()
stCoeff[0,:].reshape()
