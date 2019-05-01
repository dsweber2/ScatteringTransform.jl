import torch
from kymatio import Scattering2D
import torchvision
examples = torchvision.datasets.MNIST("~/allHail/projects/ShatteringTransform.jl/kymatioTests/",download=True)
testEx = torchvision.datasets.MNIST("~/allHail/projects/ShatteringTransform.jl/kymatioTests/",download=True,train=False)
data = torch.cat([examples.data, testEx.data])
labels =  torch.cat([examples.targets, testEx.targets])
scattering = Scattering2D(J=2, L=8, shape=(28,28))
stCoeff = scattering(data.float())
torch.save(stCoeff,"pydata.p")
stCoeff = torch.load("pydata.p")
from sklearn import svm
classifier = svm.SVC(kernel='linear',verbose=True)
from sklearn.model_selection import train_test_split
stCoeff = stCoeff.reshape(70000,-1)
dTrain,dTest, lTrain,lTest = train_test_split(stCoeff.numpy(),labels.numpy(),random_state=4025)
classifier.fit(dTrain,lTrain)
