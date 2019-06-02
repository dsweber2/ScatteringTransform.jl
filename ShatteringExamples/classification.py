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
dataSet = "MNIST"
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
        print("On " + pithyNames[iFile] + " i.e. " + str(iFile))
        classifier = svm.SVC(kernel='linear', verbose = False)
        classifier.fit(dTrain, lTrain)
        dset = resultFile.require_dataset("linearSVM/SampleRate52500",
                                          scores.shape,
                                          dtype = scores.dtype)
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





import matplotlib.pyplot as plt
import h5py
import numpy as np
import matplotlib as mplt
import matplotlib.patheffects as mpe
SizeCV = [int(np.floor(50*5**x)) for x in
          np.linspace(0,np.log(52500/50)/np.log(5),num=10)]
nRepeats = [int(np.ceil(52500/x)) for x in SizeCV[0:-1]]


baseDir = "/fasterHome/workingDataDir/shattering/"
dataSet = "FashionMNIST/"
validSets = 6
filenames = [baseDir + dataSet +"kymatResults.h5", baseDir + dataSet +
             "absResults.h5", baseDir + dataSet +"ReLUResults.h5", baseDir +
             dataSet +"tanhResults.h5", baseDir + dataSet +
             "piecewiseResults.h5", baseDir + dataSet + "softplusResults.h5"]

display_names = ["kymatio defaults", "absolute value", "ReLU", "Tanh",
                 "Piecewise Linear", "softplus"]
filenames = filenames[0:validSets]
display_names = display_names[0:validSets]
means = np.zeros((len(filenames), len(nRepeats)))
stds = np.zeros((len(filenames), len(nRepeats)))
thing = h5py.File(filenames[4])
for (i, nam) in enumerate(filenames):
    print(nam)
    with h5py.File(nam, "r") as h5File:
        for (j,x) in enumerate(nRepeats):
            samps = h5File["linearSVM/SampleRate" + str(x)]
            means[i, j] = np.mean(samps)
            stds[i, j] = np.std(samps)
with h5py.File(filenames[4], "r") as h5File:
    for (j,x) in enumerate(nRepeats):
        samps = h5File["linearSVM/SampleRate" + str(x)]
        means[4, j] = np.mean(samps)
        stds[4, j] = np.std(samps)
            
fig = plt.loglog(np.array([SizeCV[0:-1] for i in range(4)]).transpose(),
                 1-means.transpose())

mplt.style.use("seaborn-colorblind")
plt.clf()
linestyles = ['--', '-', ":", '-.', (0,(3, 1, 1, 1, 1, 1)), (0, (5,10))]
pe1 = [mpe.Stroke(linewidth=4, foreground='black'),
       mpe.Stroke(foreground='white',alpha=1),
       mpe.Normal()]
for (i, nam) in enumerate(filenames):
    curMean = 1 - means.transpose()[:,i]
    plt.fill_between(SizeCV[0:-1], curMean+stds.transpose()[:,i],
                     curMean-stds.transpose()[:,i], alpha = .75, cmap = "Set1",color='C'+str(i))
    plt.plot(SizeCV[0:-1], 1 - means.transpose()[:,i], linewidth=2, linestyle =
             linestyles[i], color='k', label = display_names[i]) 
plt.xscale('log')
plt.xticks(SizeCV[0:-1], labels=[50, 108, 234, 508, "1,100", "2,384", "5,165", "11k", "24k"])
plt.yscale('log', basey=2)
if dataSet == "MNIST/":
    yticks = [.5, .25, .1, .05, .025, .01]
else:
    yticks = [.5, .36, .25, .2, .15, .12, .1]
plt.yticks(yticks, labels = yticks)
plt.ylabel("Error rate")
plt.xlabel("Number of Training Examples")
plt.title("FashionMNIST classification CV (width is 1std)")
plt.grid(True)
plt.legend(display_names)
plt.savefig(baseDir + dataSet + "linearFashionMNIST.pdf")
plt.clf()
plt.show()
