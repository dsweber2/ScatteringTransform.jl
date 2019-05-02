a = time()
using Distributed
addprocs(5)
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
subsampling = [28/19, 19/13, 13/8]
shearLevel = Int.(ceil.((1:4)/4))
frameBound = 1.0
nonlinTypes = [(absType(),"abs"), (ReLUType(), "ReLU"), (tanhType(), "tanh"),
 (softplusType(), "softplus")]
nonlinTypes = [(softplusType(), "softplus")]
m =2

train_x, train_y = MNIST.traindata();
test_x, test_y = MNIST.testdata();
dataTot = SharedArray{Float32}(Float32.(cat(permutedims(train_x, (3,1,2)),
                                            permutedims(test_x, (3,1,2)),
                                            dims=1)));
labelTot = [train_y; test_y];

println("data loaded in $(time()-a)")
a = time()
nTot = size(dataTot, 1); 
layers = layeredTransform(m, size(dataTot)[2:3], subsamples=subsampling, shearLevel=shearLevel, typeBecomes = Float32, frameBound=frameBound)
println("successfully formed some layers in $(time()-a)")

n, q, dataSizes, outputSizes, resultingSize = ScatteringTransform.calculateSizes(layers,
                                                                                 (-1,-1),
                                                                                 (70000,28,28))
save("/fasterHome/workingDataDir/shattering/shatteredMNIST$(m).jld",
     "layers", layers)
# COLUMN-MAJOR
dimension = sum(prod(x[2:end]) for x in outputSizes)
typeBase = Float32
for nonlinTyp in nonlinTypes
    global a
    println("starting $(nonlinTyp[2])")
    filename = "/fasterHome/workingDataDir/shattering/shatteredMNIST$(nonlinTyp[2])$(m).h5"
    h5open(filename, "w") do file
        g = g_create(file, "data")
        g["labels"] = labelTot
        shatteredTraining = zeros(Float32, nTot, dimension)
        batchSize = 700
        totalBatches = ceil.(nTot/batchSize)
        start = time()
        a = time()
        for i=1:Int(totalBatches)
            global a
            println("at batch $(i), taking $(time()-a), total time of $(time()-start)")
            a = time()
            shatteredTraining[((i-1)*batchSize+1):(i*batchSize), :] =
                st(dataTot[((i-1)*batchSize+1):(i*batchSize), :, :], layers,
                   nonlinTyp[1], thin=true, verbose=false)
        end
        g["shattered"] = shatteredTraining
    end
end
