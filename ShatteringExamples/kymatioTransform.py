import torch
import h5py
import numpy as np
from kymatio import Scattering2D
import torchvision
examples = torchvision.datasets.FashionMNIST("~/allHail/projects/ShatteringTransform.jl/kymatioTests/", download=True)
testEx = torchvision.datasets.MNIST("~/allHail/projects/ShatteringTransform.jl/kymatioTests/", download=True, train=False)
data = torch.cat([examples.data, testEx.data])
labels =  torch.cat([examples.targets, testEx.targets])
labels = labels.numpy()
scattering = Scattering2D(J=2, L=8, shape=(28,28))
stCoeff = scattering(data.float())
stCoeff = stCoeff.numpy()
stCoeff = stCoeff.reshape((-1, np.prod(stCoeff.shape[1:])))
filename = "/fasterHome/workingDataDir/shattering/kymatio2FashionMNIST.h5"
with h5py.File(filename, "w") as file:
    db = file.create_group("data")
    label = db.create_dataset("labels", labels.shape, dtype=labels.dtype)
    scatterD = db.create_dataset("shattered", stCoeff.shape, dtype=stCoeff.dtype)
    scatterD[...] = stCoeff
    label[...] = labels

f = h5py.File(filename, 'r')
db = f["data"]
labels = db["labels"][:]
stCoeff = db["shattered"]
stCoeff = stCoeff[:,:].transpose()


