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

# tmp
filename = "/fasterHome/workingDataDir/shattering/kymatio2MNIST.h5"
println("On file $(filename)")
Atmp = h5open(filename, "r")
A = Atmp["data/shattered"]
size(A)
coeffs, stderrs = fitPolyDecay(A[:,4])
sort(abs.(A[:,4]'), dims = length(size(A[:,4])),rev=true)
reorderCoeff(A[:,4])
plot(log.(1:size(A,1)), sort(logg.(abs.(A[:,1]')), dims = length(size(A[:,4])),rev=true)')
savefig("tmp.pdf")
close(Atmp)

logg(x) = x==0 ? 0 : log(x)
# tmp end

reorderCoeff(A::AbstratctArray{T,2}) where T = sort(abs.(A), dims = length(size(A)),rev=true)
reorderCoeff(A::AbstratctArray{T,1}) where T = sort(abs.(A), rev=true)

"""
do a log-log fit of a matrix along the second dimension
"""
function logFit(A)
    logCoeff = log.(reorderCoeff(A))
    xAxis = 1:size(logCoeff, 2)
    
end


"""
map NaNs onto zero
"""
function getNonNans(A)
    nan_i = isnan.(A)
    return [A[map(x->!(x), nan_i)]; zeros(sum(nan_i))]
end
function summarize(coeffs)
    nan_i = isnan.(coeffs)
    meanValue = mean(coeffs[map(x->!(x), nan_i)])
    return (meanValue, sum(nan_i))
end

results = Dict((x*" "*y,summarize(Atmp[x][y]["coeffs"][:, 2])) for x in pithyNames for y in dataSets)


Atmp = h5open("/fasterHome/workingDataDir/shattering/sparsity.h5","r")


h1 = histogram(getNonNans(Atmp["ReLU/FashionMNIST/coeffs"][:,2]), norm=true,
               label="ReLU",alpha=.5, xlabel="Decay Rate",
               ylabel="Number Of Examples")
vline!((results["ReLU FashionMNIST"][1], results["ReLU FashionMNIST"][1]), label="ave")
h2 = histogram(getNonNans(Atmp["abs/FashionMNIST/coeffs"][:,2]), norm=true,
               label="abs",alpha=.5, xlabel="Decay Rate",
               ylabel="Number Of Examples")
vline!((results["abs FashionMNIST"][1], results["abs FashionMNIST"][1]), label="ave")
h3 = histogram(getNonNans(Atmp["tanh/FashionMNIST/coeffs"][:,2]), norm=true,
               label="tanh",alpha=.5, xlabel="Decay Rate",
               ylabel="Number Of Examples")
vline!((results["tanh FashionMNIST"][1], results["tanh FashionMNIST"][1]), label="ave")

h4 = histogram(getNonNans(Atmp["kymat/FashionMNIST/coeffs"][:,2]), norm=true,
               label="kymat",alpha=.5, xlabel="Decay Rate",
               ylabel="Number Of Examples")
vline!((results["kymat FashionMNIST"][1], results["kymat FashionMNIST"][1]), label="ave")

h5 = histogram(getNonNans(Atmp["piecewise/FashionMNIST/coeffs"][:,2]), norm=true,
               label="tanh",alpha=.5, xlabel="Decay Rate",
               ylabel="Number Of Examples")
vline!((results["piecewise FashionMNIST"][1], results["piecewise FashionMNIST"][1]), label="ave")
h6 = histogram(getNonNans(Atmp["softplus/FashionMNIST/coeffs"][:,2]), norm=true,
               label="softplus",alpha=.5, xlabel="Decay Rate",
               ylabel="Number Of Examples") 
vline!((results["softplus FashionMNIST"][1], results["softplus FashionMNIST"][1]), label="ave")
plot(h1, h2,h3,h4,h5,h6, layout=(2,3))
savefig("sparsityFashionMNIST.pdf")




h1 = histogram(getNonNans(Atmp["ReLU/MNIST/coeffs"][:,2]), norm=true,
               label="ReLU",alpha=.5, xlabel="Decay Rate",
               ylabel="Number Of Examples")
vline!((results["ReLU MNIST"][1], results["ReLU MNIST"][1]), label="ave")
h2 = histogram(getNonNans(Atmp["abs/MNIST/coeffs"][:,2]), norm=true,
               label="abs",alpha=.5, xlabel="Decay Rate",
               ylabel="Number Of Examples")
vline!((results["abs MNIST"][1], results["abs MNIST"][1]), label="ave")
h3 = histogram(getNonNans(Atmp["tanh/MNIST/coeffs"][:,2]), norm=true,
               label="tanh",alpha=.5, xlabel="Decay Rate",
               ylabel="Number Of Examples")
vline!((results["tanh MNIST"][1], results["tanh MNIST"][1]), label="ave")

h4 = histogram(getNonNans(Atmp["kymat/MNIST/coeffs"][:,2]), norm=true,
               label="kymat",alpha=.5, xlabel="Decay Rate",
               ylabel="Number Of Examples")
vline!((results["kymat MNIST"][1], results["kymat MNIST"][1]), label="ave")

h5 = histogram(getNonNans(Atmp["piecewise/MNIST/coeffs"][:,2]), norm=true,
               label="tanh",alpha=.5, xlabel="Decay Rate",
               ylabel="Number Of Examples")
vline!((results["piecewise MNIST"][1], results["piecewise MNIST"][1]), label="ave")
h6 = histogram(getNonNans(Atmp["softplus/MNIST/coeffs"][:,2]), norm=true,
               label="softplus",alpha=.5, xlabel="Decay Rate",
               ylabel="Number Of Examples") 
vline!((results["softplus MNIST"][1], results["softplus MNIST"][1]), label="ave")
plot(h1, h2,h3,h4,h5,h6, layout=(2,3))

savefig("sparsityMNIST.pdf")




# What does a n^(-3/2) fit give?
fDat = [i.^(-j) for i=1:10000, j=.5:.125:5]';
tmpcoeff, tmpstd = fitPolyDecay(fDat',true)
df = DataFrame(Y=log.(reorderCoeff(fDat[1,:])), X=log.(1:size(fDat,2)))
ols = lm(@formula(Y~X), df)


# What does a log(n)^2/n^(-3/2) graph give?
fDat = [(log(i)/i).^(j) for i=2:10000, j=.5:.125:5]
tmpcoeff, tmpstd = fitPolyDecay(fDat)
df = DataFrame(Y=log.(reorderCoeff(fDat[1,:])), X=log.(1:size(fDat, 2)))
ols = lm(@formula(Y~X), df)
