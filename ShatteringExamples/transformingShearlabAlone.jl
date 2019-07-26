#using Distributed
#addprocs(13)
using MLDatasets
using MLDataUtils
using HDF5
using Shearlab
#@everywhere using ScatteringTransform
using JLD
using SharedArrays
#@everywhere using FFTW
using FFTW




println("using statements done")
train_x, train_y = FashionMNIST.traindata();
test_x, test_y = FashionMNIST.testdata();
dataTot = SharedArray{Float32}(Float32.(cat(train_x, test_x, dims=3)));
labelTot = [train_y; test_y];
file = open("/fasterHome/workingDataDir/shattering/ShearLab_FashionMNIST_labels_juliaOrdered.bin", "w+")
write(file, length(labelTot))
write(file, labelTot)
close(file)
println("data loaded")
#@everywhere nTot = 70000; @everywhere m =2
nTot = 70000; m =2
#@everywhere layers = layeredTransform(m, (28, 28), subsamples=[28/19, 19/13,13/8],shearLevel=Int.(ceil.((1:4)/4)),typeBecomes = Float32);
shears = Shearlab.getshearletsystem2D(28, 28, 2,typeBecomes=Float32);
@time firstTen = Shearlab.sheardec2D(dataTot[:,:,1:1000], shears);
using Plots
function compareShearLoc(shears, transformed, original, shearNum, nExample)
    baseDomainShearlet = irfft(ifftshift(shears.shearlets[:,:,shearNum]), shears.padBy[1]*2 + size(transformed,1))
    plot(heatmap(transformed[:,:,shearNum,nExample],title="Sheared"), heatmap(baseDomainShearlet, title="Shearlet"), heatmap(original[:,:,nExample], title="Original"))
end
netResults = Shearlab.sheardec2D(dataTot, shears);
s = h5open("/fasterHome/workingDataDir/shattering/shearlab_FashionMNIST_2shearLevels.h5", "cw")
A = reshape(tmp[:,:,:,:], (prod(size(tmp)[1:3]), size(tmp,4)))
(prod(size(tmp)[1:3]), size(tmp,4))
tmp = s["data/shattered"]
s["data/shattered"] = netResults
close(s)
s = open("/fasterHome/workingDataDir/shattering/shearlab_FashionMNIST_2shearLevels.bin", "w")
# We'll write the dimensions of the array as the first two Ints in the file
write(s, 28*28*size(shears.shearlets, 3))
write(s, nTot)
write(s, reshape(netResults, (:,nTot)))
close(s)
println("I'm all done now how 'bout you?")
