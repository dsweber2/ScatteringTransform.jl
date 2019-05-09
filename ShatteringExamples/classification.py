import numpy as np
import h5py

# import the data
source_file = "/fasterHome/workingDataDir/shattering/shatteredMNISTabs2.h5"
# source_file = "/fasterHome/workingDataDir/shattering/kymatioMNIST.h5"
# load the data
f = h5py.File(source_file, 'r')
db = f["data"]
labels = db["labels"][:]
stCoeff = db["shattered"]
if "shattered" in source_file:
    stCoeff = stCoeff[:,:].transpose()

from sklearn import svm

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

dTrain, dTest, lTrain, lTest = train_test_split(stCoeff, labels,
                                                random_state=4025)


startPoint = 0
# resultFile = h5py.File("/fasterHome/workingDataDir/shattering/MNIST/kymatResults.h5", "a")
resultFile = h5py.File("/fasterHome/workingDataDir/shattering/MNIST/results.h5", "a")
resgrp = resultFile.require_group("linearSVM")

# set the size of the training set via setting the cv number of folds
N = lTrain.shape[0]
SizeCV = [int(np.floor(50*5**x)) for x in
          np.linspace(0,np.log(dTrain.shape[0]/50)/np.log(5),num=10)]
nRepeats = [int(np.ceil(N/x)) for x in SizeCV[startPoint:-1]]
nRepeats.reverse()
x = nRepeats[-1]
classifiers = [svm.SVC(kernel='linear', verbose=False) for x in nRepeats]

for (i,x) in enumerate(nRepeats):
    print("On " + str(x) + " i.e. " + str(i) + " out of " + str(len(nRepeats)))
    repeater = StratifiedKFold(x)
    reRepeater = repeater.split(dTrain, lTrain)
    flippedFolder = ((y[1], y[0]) for y in reRepeater) # a generator with large test sets and small train sets
    scores = cross_val_score(classifiers[i], dTrain, lTrain, cv=flippedFolder, n_jobs=12)
    dset = resultFile.require_dataset("linearSVM/SampleRate" + str(x), scores.shape,
                             dtype = scores.dtype)
    dset[:] = scores
    resultFile.flush()
resultFile.close()
dset = resultFile.create_dataset("linearSVM/SampleRate" + str(nRepeats[0]), scores.shape,
                             dtype = scores.dtype)
resultFile["linearSVM/SampleRate" + str(nRepeats[0])]
dset[:] = resultFile["results/SampleRate" + str(nRepeats[0])]
tmpResults = resultFile["results/SampleRate" + str(nRepeats[0])]
import pickle
f = open("/fasterHome/workingDataDir/shattering/MNIST/kymatClassifiers.p","wb")
pickle.dump(classifiers,f)

f = open("/fasterHome/workingDataDir/shattering/MNIST/classifiers.p","r")
pickle.load(f)
import matplotlib.pyplot as plt
plt.plot(classifier.support_vectors_)
plt.show()
stCoeff[0,:].reshape()

import h5py
resultFile = h5py.File("/fasterHome/workingDataDir/shattering/MNIST/kymatResults.h5", "r+")





SizeCV = [int(np.floor(50*5**x)) for x in
          np.linspace(0,np.log(52500/50)/np.log(5),num=10)]
nRepeats = [int(np.ceil(52500/x)) for x in SizeCV[0:-1]]
