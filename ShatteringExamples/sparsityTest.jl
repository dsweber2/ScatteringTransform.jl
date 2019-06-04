# using ScatteringTransform 
using Plots, Statistics, PowerLaws, MLDatasets
using Distributed
@everywhere using HDF5

@everywhere function reorderCoeff(A::AbstractArray{T,2}) where T
    if size(A,1)==1
        includingZeros = reshape(sort(abs.(A), dims =
                                      length(size(A)),rev=true), (length(A,)))
    else
        includingZeros = sort(abs.(A), dims = length(size(A)),rev=true)
    end
    firstZero = findfirst(includingZeros.==0)
    if typeof(firstZero)<:Nothing
        return includingZeros
    else
        return includingZeros[1:firstZero]
    end
end
@everywhere reorderCoeff(A::AbstractArray{T,1}) where T = sort(abs.(A), rev=true)

"""
    conPowerFit, KSstatistic = testSingleRow(A, i)
given an HDF5 pointer A and an example index i, compute estimate_xmin on the single row i
"""
function testSingleRow(A, i; frac = 1.0)
    keep = floor(Int, frac*size(A,2))
    thing = reshape(reorderCoeff(A[i, :])[1:keep], (keep,))
    if length(thing)>1
        return estimate_xmin(thing, con_powerlaw)
    end
end

"""
     (conPowers, KStestPValues, αs, θs) = testWholematrix(A)

given an HDF5 pointer A, compute estimate_xmin on every row separately
"""
function testWholeMatrix(A; frac=1.0)
    αs = zeros(size(A,1))
    θs = zeros(size(A,1))
    conPowers = Array{con_powerlaw}(undef, size(A,1))
    KStestPValues = zeros(size(A,1))
    for i = 1:size(A, 1)
        conPowers[i], KStestPValues[i] = testSingleRow(A, i; frac=frac)
        αs[i] = conPowers[i].α
        θs[i] = conPowers[i].θ
        if i%100==0
            println("On $(i)")
        end
    end
    return (conPowers, KStestPValues, αs, θs)
end
# load the dataset; row examples (nExamples×dims)


using GLM, DataFrames, Distributed, SharedArrays
addprocs(10)
@everywhere using GLM, DataFrames, SharedArrays
function fitPolyDecay(A)
    coeffs = SharedArray{Float64}(zeros(size(A,1), 2))
    stderrs = SharedArray{Float64}(zeros(size(A,1), 2))
    for i=1:size(A,1)
        df = DataFrame(X=Float64.(log.(1:size(A,2))), Y=Float64.(log.(reorderCoeff(A[i,:]))))
        ols = lm(@formula(Y~X), df)
        coeffs[i,:] = coef(ols) .*[1, -1]
        stderrs[i,:] = stderror(ols)
        if i%100==0
            println("On $(i)")
        end
    end
    return (coeffs, stderrs)
end
function fitPolyDecay(A,transpose=true)
    coeffs = SharedArray{Float64}(zeros(size(A,2), 2))
    stderrs = SharedArray{Float64}(zeros(size(A,2), 2))
    for i=1:size(A,2)
        reorderedData = Float64.(log.(reorderCoeff(A[:, i]')))
        df = DataFrame(X=Float64.(log.(1:size(reorderedData,1))),
                       Y=reorderedData)
        ols = lm(@formula(Y~X), df)
        coeffs[i,:] = coef(ols) .*[1, -1]
        stderrs[i,:] = stderror(ols)
        if i%100==0
            println("On $(i)")
        end
    end
    return (coeffs, stderrs)
end

function summarize(coeffs, i=1)
    nan_i = isnan.(coeffs[:,i])
    meanValue = mean(coeffs[map(x->!(x), nan_i), i])
    return 
end
pithyNames = ["abs", "ReLU", "tanh", "softplus", "piecewise", "kymat"]
dataSets = ["FashionMNIST", "MNIST"]
filenames = ["/fasterHome/workingDataDir/shattering/shattered"*y*x*"2.h5"
             for x in pithyNames for y in dataSets]
# filename =  "/fasterHome/workingDataDir/shattering/shatteredMNISTpiecewise2.h5"
# filename = "/fasterHome/workingDataDir/shattering/kymatio2MNIST.h5"
# Atmp = h5open(filename,"r")
# A = Atmp["data/shattered"]
writeTo = h5open("/fasterHome/workingDataDir/shattering/sparsity.h5","w")
for pith in pithyNames, dat in dataSets
    if pith=="kymat"
        filename = "/fasterHome/workingDataDir/shattering/kymatio2"*dat*".h5"
    else
        filename = "/fasterHome/workingDataDir/shattering/shattered"*dat*pith*"2.h5"
    end
    println("On file $(filename)")
    Atmp = h5open(filename, "r")
    A = Atmp["data/shattered"]
    coeffs, stderrs = fitPolyDecay(A)
    close(Atmp)
    try
        writeTo["$(pith)/$(dat)/coeffs"] = coeffs
        writeTo["$(pith)/$(dat)/stderrs"] = stderrs
    catch
        println("already wrote $pith/$dat")
    end
end
error("All done now")
for filename in filenames
    println("On file $(filename)")
end

Atmp = h5open(filenames[4], "r+")
sum(isnan.(Atmp["sparsity/coeffs"][:,2]))
close(Atmp)
train_x, train_y = MNIST.traindata();
test_x, test_y = MNIST.testdata();
dataTot = SharedArray{Float32}(Float32.(cat(permutedims(train_x, (3,2,2)),
                                            permutedims(test_x, (3,1,2)),
                                            dims=1)))
A = reshape(dataTot, (size(dataTot, 1), :))
filename = "/fasterHome/workingDataDir/shattering/sparsityMNIST.h5"
coeffs, stderrs = fitPolyDecay(A)
getAveNonNan(coeffs,2)
Atmp = h5open(filename,"w")
Atmp["sparsity/coeffs"] = coeffs
Atmp["sparsity/stderrs"] = stderrs
close(Atmp)
plot(reorderA)

fDat = [i.^(-j) for i=1:10000, j=.5:.125:5]'
tmpcoeff, tmpstd = fitPolyDecay(fDat)
df = DataFrame(Y=log.(reorderCoeff(fDat[1,:])), X=log.(1:size(fDat,2)))
ols = lm(@formula(Y~X), df)



reorderCoeff(A::AbstratctArray{T,2}) where T = sort(abs.(A), dims = length(size(A)),rev=true)
reorderCoeff(A::AbstratctArray{T,1}) where T = sort(abs.(A), rev=true)
"""
do a log-log fit of a matrix along the second dimension
"""
function logFit(A)
    logCoeff = log.(reorderCoeff(A))
    xAxis = 1:size(logCoeff, 2)
    
end
