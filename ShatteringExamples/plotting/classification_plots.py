import matplotlib.pyplot as plt
import h5py
import numpy as np
import matplotlib as mplt
import matplotlib.patheffects as mpe
SizeCV = [int(np.floor(50*5**x)) for x in
          np.linspace(0,np.log(52500/50)/np.log(5),num=10)]
nRepeats = [int(np.ceil(52500/x)) for x in SizeCV[0:-1]]
nRepeats.append(52500)

baseDir = "/fasterHome/workingDataDir/shattering/"
dataSet = "MNIST/"
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
        print(list(h5File["linearSVM"].keys()))
        for (j,x) in enumerate(nRepeats):
            samps = h5File["linearSVM/SampleRate" + str(x)]
            means[i, j] = np.mean(samps)
            stds[i, j] = np.std(samps)

            

mplt.style.use("seaborn-colorblind")
plt.clf()
linestyles = ['--', '-', ":", '-.', (0,(3, 1, 1, 1, 1, 1)), (0, (5,10))]
pe1 = [mpe.Stroke(linewidth=4, foreground='black'),
       mpe.Stroke(foreground='white',alpha=1),
       mpe.Normal()]
for (i, nam) in enumerate(filenames):
    curMean = 1 - means.transpose()[:,i]
    plt.fill_between(SizeCV, curMean+stds.transpose()[:,i],
                     curMean-stds.transpose()[:,i], alpha = .75, cmap = "Set1",color='C'+str(i))
    plt.plot(SizeCV, 1 - means.transpose()[:,i], linewidth=2, linestyle =
             linestyles[i], color='k', label = display_names[i]) 
plt.xscale('log')
plt.xticks(SizeCV, labels=[50, 108, 234, 508, "1,100", "2,384", "5,165", "11k", "24k", "52.5k"])
plt.yscale('log', basey=2)
if dataSet == "MNIST/":
    yticks = [.5, .25, .1, .05, .025, .01]
else:
    yticks = [.5, .36, .25, .2, .15, .12, .1]
plt.yticks(yticks, labels = yticks)
plt.ylabel("Error rate")
plt.xlabel("Number of Training Examples")
plt.title("MNIST classification CV (width is 1std)")
plt.grid(True)
plt.legend(display_names)
plt.savefig(baseDir + dataSet + "linearMNIST.pdf")
plt.clf()
plt.show()


import pickle
import h5py
dataSets = ["MNIST", "FashionMNIST"]
pithyNames = ["abs", "ReLU", "tanh", "softplus", "piecewise"]
pithyNames = ["piecewise"]
baseDir = "/fasterHome/workingDataDir/shattering/"

iFile = 0
dataSeti = 0
for dataSeti in range(2):
dataSet = dataSets[dataSeti]
    resultFile = h5py.File(baseDir + dataSet + "/" + "finalCoeffs.h5", "a")
    for iFile in range(len(pithyNames)):
        fileObject = open(baseDir + dataSet + "/" + pithyNames[iFile] +
                          "finalClassifier.p", 'rb')
        thing = pickle.load(fileObject)
        resultFile[pithyNames[iFile]] = thing.coef_
        fileObject.close()
    resultFile.close()

dataSet = dataSets[0]
resultFile = h5py.File(baseDir + dataSet + "/" + "finalCoeffs.h5", "a")
fileObject = open(baseDir + dataSet + "/" + pithyNames[0] +
                  "finalClassifier.p", 'rb')
thing = pickle.load(fileObject)
resultFile[pithyNames[0]] = thing.coef_
fileObject.close()
resultFile.close()

thing
