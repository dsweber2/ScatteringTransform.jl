a = time()
using Distributed
addprocs(2)
using MLDatasets
using MLDataUtils
using HDF5
@everywhere using ScatteringTransform
using JLD
using SharedArrays
@everywhere using FFTW
using FFTW
println("using statements done in $(time()-a)")
a = time()


whichset = 4
whichNonlin = 5
m = 2


if maximum(whichset.==[1,4])
    subsampling = [28/19, 19/13, 13/8]
elseif whichset==2
    subsampling = [32/25, 25/14, 14/10]
end
shearLevel = Int.(ceil.((1:4)/4))
frameBound = 1.0
nonlinTypes = [(absType(),"abs"), (ReLUType(), "ReLU"), (tanhType(), "tanh"),
 (softplusType(), "softplus"), (piecewiseType(), "piecewise")]

datasets = [(MNIST,"MNIST"), (CIFAR10, "CIFAR10"), (CIFAR100, "CIFAR100"), (FashionMNIST,"FashionMNIST")]

nonlinTypes = [nonlinTypes[whichNonlin]]


train_x, train_y = datasets[whichset][1].traindata();
test_x, test_y = datasets[whichset][1].testdata();
if maximum(whichset .== [1, 4])
    dataTot = SharedArray{Float32}(Float32.(cat(permutedims(train_x, (3,1,2)),
                                                permutedims(test_x, (3,1,2)),
                                                dims=1)));
elseif whichset == 2
    dataTot = SharedArray{Float32}(Float32.(cat(permutedims(train_x, (4,3,1,2)),
                                                permutedims(test_x, (4,3,1,2)),
                                                dims=1)));
end
labelTot = [train_y; test_y];

println("data loaded in $(time()-a)")
a = time()
nTot = size(dataTot, 1); 
layers = layeredTransform(m, size(dataTot)[end-1:end], subsamples=subsampling, shearLevel=shearLevel, typeBecomes = Float32, frameBound=frameBound)
println("successfully formed some layers in $(time()-a)")

n, q, dataSizes, outputSizes, resultingSize = ScatteringTransform.calculateSizes(layers,
                                                                                 (-1,-1),
                                                                                 (70000,28,28))
save("/fasterHome/workingDataDir/shattering/shattered$(datasets[whichset][2])$(m).jld",
     "layers", layers)
# COLUMN-MAJOR
dimension = sum(prod(x[2:end]) for x in outputSizes)
typeBase = Float32
innerAxes = axes(dataTot)[2:end]
for nonlinTyp in nonlinTypes
    global a
    println("starting $(nonlinTyp[2])")
    filename = "/fasterHome/workingDataDir/shattering/shattered$(datasets[whichset][2])$(nonlinTyp[2])$(m).h5"
    h5open(filename, "w") do file
        g = g_create(file, "data")
        g["labels"] = labelTot
        g["shattered"] = zeros(Float32, size(dataTot)[1:end-2]..., dimension)
        systemCopy = readmmap(g["shattered"])
        outerAxes = axes(systemCopy)
        batchSize = 700
        totalBatches = ceil.(nTot/batchSize)
        start = time()
        a = time()
        for i=1:Int(totalBatches)
            global a
            println("at batch $(i), taking $(time()-a), total time of $(time()-start)")
            a = time()
            systemCopy[((i-1)*batchSize+1):(i*batchSize), outerAxes[2:end]...] =
                st(dataTot[((i-1)*batchSize+1):(i*batchSize), innerAxes...], layers,
                   nonlinTyp[1], thin=true, verbose=false)
        end
    end
end


# using HDF5
# f = h5open("tmp.h5","w")
# f["tmp/data"] = zeros(100,100)
# f["tmp/massive"] = zeros(100000,10000)
# wef = 3
# readmmap(f["tmp/massive"])
# f["tmp/data"][:,200] = randn(100)
# f["tmp/data"][:,1]
# [:,1]
# = randn(100)
