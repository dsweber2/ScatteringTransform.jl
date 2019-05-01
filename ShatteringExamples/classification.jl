

@sk_import linear_model: 


s = open("/fasterHome/workingDataDir/shattering/shatteredMNISTsoftplus2_4shearLevels.bin", "r")
dimension = read(s,Int)
nTot = read(s,Int)
transData = read!(s,zeros(Float32, (dimension, nTot)))




from sklearn import svm
classifier = svm.SVC(kernel='linear',verbose=True)
from sklearn.model_selection import train_test_split
stCoeff = stCoeff.reshape(70000,-1)
dTrain,dTest, lTrain,lTest = train_test_split(stCoeff.numpy(),labels.numpy(),random_state=4025)
classifier.fit(dTrain,lTrain)
