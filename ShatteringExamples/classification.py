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

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

dTrain, dTest, lTrain, lTest = train_test_split(stCoeff, labels,
                                                random_state=4025)


startPoint = 0
resultFile = h5py.File("/fasterHome/workingDataDir/shattering/MNIST/results.h5", "a")
resgrp = resultFile.require_group("linearSVM")

# set the size of the training set via setting the cv number of folds
SizeCV = [int(np.floor(50*5**x)) for x in
          np.linspace(0,np.log(dTrain.shape[0]/50)/np.log(5),num=10)]
nRepeats = [int(np.ceil(N/x)) for x in SizeCV[startPoint:-1]]
nRepeats.reverse()
N = lTrain.shape[0]
x = nRepeats[-1]
classifiers = [svm.SVC(kernel='linear', verbose=True) for x in nRepeats]

for (i,x) in enumerate(nRepeats[2:]):
    print("On " + str(x) + " i.e. " + str(i))
    repeater = StratifiedKFold(x)
    reRepeater = repeater.split(dTrain, lTrain)
    flippedFolder = ((y[1], y[0]) for y in reRepeater) # a generator with large test sets and small train sets
    scores = cross_val_score(classifiers[i], dTrain, lTrain, cv=flippedFolder)
    dset = resultFile.require_dataset("linearSVM/SampleRate" + str(x), scores.shape,
                             dtype = scores.dtype)
    dset[:] = scores
    resultFile.flush()
dset = resultFile.create_dataset("linearSVM/SampleRate" + str(nRepeats[0]), scores.shape,
                             dtype = scores.dtype)
resultFile["linearSVM/SampleRate" + str(nRepeats[0])]
dset[:] = resultFile["results/SampleRate" + str(nRepeats[0])]
tmpResults = resultFile["results/SampleRate" + str(nRepeats[0])]
import pickle
pickle.dump(classifiers,f)
f = open("/fasterHome/workingDataDir/shattering/MNIST/results.h5","wb")

import matplotlib.pyplot as plt
plt.plot(classifier.support_vectors_)
plt.show()
stCoeff[0,:].reshape()
