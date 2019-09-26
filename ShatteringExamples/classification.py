import numpy as np
import h5py
import pickle
from sklearn import svm


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline        import Pipeline
from sklearn.decomposition   import PCA

# import the data
dataSet = "FashionMNIST"
pithyNames = ["abs", "ReLU", "tanh", "softplus", "piecewise"]
baseDir = "/fasterHome/workingDataDir/shattering/"
source_files = [baseDir + "shattered" + dataSet + x + "2.h5" for x in pithyNames]
res_files = [baseDir + dataSet + "/" + x + "Results.h5" for x in pithyNames]

# source_files = ["/fasterHome/workingDataDir/shattering/kymatio2FashionMNIST.h5", "/fasterHome/workingDataDir/shattering/kymatio2MNIST.h5"]
# res_files = ["/fasterHome/workingDataDir/shattering/FashionMNIST/kymatResults.h5", "/fasterHome/workingDataDir/shattering/MNIST/kymatResults.h5"]
#source_files = ["/fasterHome/workingDataDir/shattering/shatteredFashionMNISTpiecewise2.h5", "/fasterHome/workingDataDir/shattering/shatteredMNISTpiecewise2.h5"]
# res_files = ["/fasterHome/workingDataDir/shattering/FashionMNIST/piecewiseResults.h5", "/fasterHome/workingDataDir/shattering/MNIST/piecewiseResults.h5"]

final = True

# load the data
for (iFile, source_file) in enumerate(source_files):
    res_file = res_files[iFile]
    f = h5py.File(source_file, 'r')
    db = f["data"]
    labels = db["labels"][:]
    stCoeff = db["shattered"][:,:]
    f.close()
    if "shattered" in source_file:
        stCoeff = stCoeff[:,:].transpose()
    dTrain, dTest, lTrain, lTest = train_test_split(stCoeff, labels,
                                                    random_state=4025)

    startPoint = 0
    resultFile = h5py.File(res_file, "a")

    # set the size of the training set via setting the cv number of folds
    N = lTrain.shape[0]
    SizeCV = [int(np.floor(50*5**x)) for x in
              np.linspace(0,np.log(dTrain.shape[0]/50)/np.log(5),num=10)]
    nRepeats = [int(np.ceil(N/x)) for x in SizeCV[startPoint:-1]]
    nRepeats.reverse()
    SizeCV.reverse()

    if final:
        print("Running Final, On " + pithyNames[iFile] + " i.e. " + str(iFile))
        classifier = svm.SVC(kernel='linear', verbose = False)
        classifier.fit(dTrain, lTrain)
        dset = resultFile.require_dataset("linearSVM/SampleRate52500",
                                          (1,),
                                          dtype = np.dtype('float64'))
        dset[:] = classifier.score(dTest, lTest)
        fileObject = open(baseDir + dataSet + "/" + pithyNames[iFile] +
                          "finalClassifier.p", 'wb')
        pickle.dump(classifier, fileObject)
    else:
        for (i,x) in enumerate(nRepeats):
            print("On " + str(x) + " i.e. " + str(i) + " out of " +
                  str(len(nRepeats)))
            repeater = StratifiedKFold(x)
            reRepeater = repeater.split(dTrain, lTrain)
            flippedFolder = ((y[1], y[0]) for y in reRepeater) # a generator
            # with large test sets and small train sets
            classifier = svm.SVC(kernel='linear', verbose = False)
            scores = cross_val_score(classifier, dTrain, lTrain,
                                     cv=flippedFolder, n_jobs=2)
            dset = resultFile.require_dataset("linearSVM/SampleRate" + str(x),
                                              scores.shape,
                                              dtype = scores.dtype)
            dset[:] = scores
            resultFile.flush()
        print("finished " + str(iFile))
        resultFile.close()


