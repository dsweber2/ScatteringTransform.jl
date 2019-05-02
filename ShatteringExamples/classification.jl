using Plots

@sk_import svm : svc


s = open("/fasterHome/workingDataDir/shattering/shatteredMNISTabs2_4shearLevels.bin", "r")
dimension = read(s,Int)
nTot = read(s,Int)
transData = read!(s,zeros(Float32, (dimension, nTot)))

using ScatteringTransform
m=2;
layers = layeredTransform(m, (28, 28), subsamples=[28/19, 19/13, 13/8], shearLevel=Int.(ceil.((1:4)/4)), typeBecomes = Float32, frameBound=1.0)
results = ScatteringTransform.wrap(layers, transData[:,1],zeros(28,28))
heatmap(results.output[1][:,:,1])


s = open("/fasterHome/workingDataDir/shattering/shatteredMNISTSmallTest.bin", "w+")
write(s, 20)
write(s, 10340)
write(s, shatteredTraining)
write(s, shatteredTraining)
close(s)

s = open("/fasterHome/workingDataDir/shattering/shatteredMNISTSmallTest.bin", "r")
dimension = read(s,Int)
nTot = read(s,Int)
transData = read!(s,zeros(Float32, (20, 10340)))
close(s)

filename = "/fasterHome/workingDataDir/shattering/shatteredMNISTSmallTest.h5"
h5open(filename, "w") do file
    g = g_create(file, "data")
    attrs(g)["nTot"] = nTot
    attrs(g)["dimension"] = dimension
end

stDict = h5open(filename, "r") do file
    read(file,"data")
end
stCoeffs = stDict["d1"]

from sklearn import svm
classifier = svm.SVC(kernel='linear',verbose=True)
from sklearn.model_selection import train_test_split
stCoeff = stCoeff.reshape(70000,-1)
dTrain,dTest, lTrain,lTest = train_test_split(stCoeff.numpy(),labels.numpy(),random_state=4025)
classifier.fit(dTrain,lTrain)
