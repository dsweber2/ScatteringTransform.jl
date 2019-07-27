using ScatteringTransform 
using Plots, Statistics, PowerLaws, MLDatasets
using HDF5

function reorderCoeff(A::AbstractArray{T,2}) where T
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
reorderCoeff(A::AbstractArray{T,1}) where T = sort(abs.(A), rev=true)

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


using GLM, DataFrames, Distributed, SharedArrays, Plots
addprocs(10)
@everywhere using GLM, DataFrames, SharedArrays
function fitPolyDecay(A)
    println(size(A))
    coeffs = zeros(3, size(A,2))
    stderrs = zeros(3, size(A,2))
    plts = Array{Any}(undef, ceil(Int,size(A,2)/100))
    for i=1:size(A,2)
        X = Float64.(log.(1 .+(1:size(A,1))))
        df = DataFrame(X=X, lnX=log.(X),
                       Y=Float64.(log.(reorderCoeff(A[:, i]))))
        ols = lm(@formula(Y~ X + lnX), df)
        coeffs[:, i] = coef(ols) .*[1, -1, -1]
        stderrs[:, i] = stderror(ols)
        if i%100==0
            println("On $(i), $(floor(Int, i/100))")
            plts[floor(Int, i/100)] = plot(df[:X], df[:Y])
        end
    end
    return (coeffs, stderrs, plts)
end

coeffs, stderrs, plts = fitPolyDecay(A[:,1:1000])
plts[5]
coeffs
function fitPolyDecay(A,transpose=true)
    coeffs = SharedArray{Float64}(zeros(2, size(A,1)))
    stderrs = SharedArray{Float64}(zeros(2, size(A,1)))
    for i=1:size(A,1)
        reorderedData = Float64.(log.(reorderCoeff(A[i, :]')))
        df = DataFrame(X=Float64.(log.(1:size(reorderedData,1))),
                       Y=reorderedData)
        ols = lm(@formula(Y~X), df)
        coeffs[:, i] = coef(ols) .*[1, -1]
        stderrs[:, i] = stderror(ols)
        if i%100==0
            println("On $(i)")
        end
    end
    return (coeffs, stderrs)
end


#########################################################################################################
# comparing the actual decay paths of Kymat and shearlets
dataset = 2
pithyNames = ["shearlab"]
dataSets = ["FashionMNIST", "MNIST"]
filenames = ["/fasterHome/workingDataDir/shattering/"*x*"_"*y*"_2shearLevels.h5"
             for x in pithyNames for y in dataSets]
Atmp = h5open(filenames[dataset], "r")
A = Atmp["data/shattered"][:,:,:,:];
A = reshape(A, (prod(size(A)[1:3]), size(A,4)));
Btmp = h5open("/fasterHome/workingDataDir/shattering/kymatio2$(dataSets[dataset]).h5")
B = Btmp["data/shattered"][:,:];
Ctmp = h5open("/fasterHome/workingDataDir/shattering/shattered$(dataSets[dataset])abs2.h5")
C = Ctmp["data/shattered"][:,:];
coeffsA, stderrsA, pltsA = fitPolyDecay(A[:,1:1000])
coeffsB, stderrsB, pltsB = fitPolyDecay(A[:,1:1000])
coeffsC, stderrsC, pltsC = fitPolyDecay(C[:,1:1000])
using Statistics, Plots
reordA = cat([reorderCoeff(A[:,i]) for i=1:1000]..., dims=2);
shearAve = mean(reordA, dims=2);
shearstd = std(reordA, dims=2);
reordB = cat([reorderCoeff(B[:,i]) for i=1:1000]..., dims=2);
kymatAve = mean(reordB,dims=2);
kymatstd = std(reordB, dims=2);
reordC = cat([reorderCoeff(C[:,i]) for i=1:1000]..., dims=2);
shatterAve = mean(reordC, dims=2);
shatterstd = std(reordC, dims=2);
∇choice = ColorGradient(:dense)

function linearEstimator(y, X)
    intercept = mean(X)
    X = X .- intercept
    slope = (X'*X) \(X'*y)
    return (slope[1], intercept)
end
function linEst(y, X)
    slope,inter = linearEstimator(y, X)
    return X*slope .+ inter
end
function genloglin(coeffs,X)
    coeffs[1] .- (X .* coeffs[2] + log.(X) .* coeffs[3])
end
linEst(log.(kymatAve), log.(1:size(B,1)))
linearEstimator(log.(kymatAve), log.(1:size(B,1)))
coeffsB, stderrsB, plts = fitPolyDecay(kymatAve)
plot(Xloc(B), genloglin(coeffsB,Xloc(B)))
plot(Xloc(B), [genloglin(coeffsB,Xloc(B)) log.(kymatAve)])
coeffs, stderrs, plts = fitPolyDecay(shearAve)
coeffs, stderrs, plts = fitPolyDecay(shatterAve)

c = .3; plot(log.(1:size(B,1)), log.(kymatAve), color=∇choice[c],
             label="kymat first 1000 ave")
plot!(log.(1:size(B,1)), log.(kymatAve)-log.(kymatstd), alpha=.5,
      color=∇choice[c], fillcolor=∇choice[c],
      fillrange=log.(kymatAve)+log.(kymatstd),
      label="kymat first 1000 std Dev", legend=:bottomleft)
# plot!(log.(1:size(B,1)), linEst(log.(kymatAve), log.(1:size(B,1))),
#       color=∇choice[c])

c = .7; plot!(log.(1:size(A,1)), log.(shearAve), color=∇choice[c],
              label="shear ave")
plot!(log.(1:size(A,1)), log.(shearAve)-log.(shearstd), alpha=.5,
      color=∇choice[c], fillcolor=∇choice[c],
      fillrange=log.(shearAve)+log.(shearstd),
      label="shear std Dev",legend=:bottomleft) 
# plot!(log.(1:size(A,1)), linEst(log.(shearAve), log.(1:size(A,1))),
#       color=∇choice[c])

c = .9; plot!(log.(1:size(C,1)), log.(shatterAve), color=∇choice[c],
              label="shatter ave")
plot!(log.(1:size(C,1)), log.(shatterAve)-log.(shatterstd), alpha=.5,
      color=∇choice[c], fillcolor=∇choice[c],
      fillrange=log.(shatterAve)+log.(shatterstd),
      label="shatter std Dev",legend=:bottomleft) 
# plot!(log.(1:size(C,1)), linEst(log.(shatterAve), log.(1:size(C,1))),
#       color=∇choice[c], label="linear fit shatter")
ylims!((-15,10))
title!("Comparison of decay rates for the first 1000 MNIST images:\n Kymatio, Shearlab, and shattering")
xlabel!("log(index)")
ylabel!("log(value)")
savefig("DecayRatesPlotMNIST.pdf")

plot(log.(1:size(B,1)), log.(reorderCoeff(B[:,1])),label="prototypical")
plot!(log.(1:size(A,1)), log.(reorderCoeff(A[:,1])))
plot(log.(1:size(A,1)), max.(-22, [log.(reorderCoeff(A[:,1])) log.(reorderCoeff(A[:,2])) -coeffsA[2,2]*log.(1:size(A,1)) .+ coeffsA[1,2]]), labels = ["example with Inf slope", "ex with given fit", "fit"], xlabel ="log(index)", ylabel="log(value)")
plot(log.(1:size(A,1)),log.(reorderCoeff(A[:,2])))

#########################################################################################################
#########################################################################################################




pithyNames = ["shearlab"]
dataSets = ["FashionMNIST", "MNIST"]
filenames = ["/fasterHome/workingDataDir/shattering/"*x*"_"*y*"_2shearLevels.h5"
             for x in pithyNames for y in dataSets]
writeTo = h5open("/fasterHome/workingDataDir/shattering/sparsity.h5","cw")
names(writeTo)
for pith in pithyNames, dat in dataSets
    if pith=="kymat"
        filename = "/fasterHome/workingDataDir/shattering/kymatio2"*dat*".h5"
    else
        filename = "/fasterHome/workingDataDir/shattering/"*pith*"_"*dat*"_2shearLevels.h5"
    end
    println("On file $(filename)")
    Atmp = h5open(filename, "r")
    A = Atmp["data/shattered"][:,:,:,:]
    A = reshape(A, (prod(size(A)[1:3]), size(A,4)))
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
results = Dict((x*" "*y,summarize(Atmp[x][y]["coeffs"][2, :])) for x in pithyNames for y in dataSets)

# Atmp = h5open("/fasterHome/workingDataDir/shattering/sparsity.h5","cw")
# names(Atmp)

# Atmp2 = h5open("/VastExpanse/data/shattering/sparsity.h5","r")
# for folder in names(Atmp2)
#     for dataset in names(Atmp2[folder])
#         for dType in names(Atmp2[joinpath(folder, dataset)])
#             #println(Atmp2[joinpath(folder, dataName)])
#             println(size(Atmp2[joinpath(folder, dataset, dType)]))
#         end
#     end
# end
# Atmp[]
Atmp = writeTo
using Plots
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
               label="piecewise",alpha=.5, xlabel="Decay Rate",
               ylabel="Number Of Examples")
vline!((results["piecewise FashionMNIST"][1], results["piecewise FashionMNIST"][1]), label="ave")
h6 = histogram(getNonNans(Atmp["softplus/FashionMNIST/coeffs"][:,2]), norm=true,
               label="softplus",alpha=.5, xlabel="Decay Rate",
               ylabel="Number Of Examples") 
vline!((results["softplus FashionMNIST"][1], results["softplus
               FashionMNIST"][1]), label="ave")
h7 = histogram!(getNonNans(Atmp["shearlab/FashionMNIST/coeffs"][2,:]), norm=true,
               label="shearlab",alpha=.5, xlabel="Decay Rate",
               ylabel="Number Of Examples")
vline!((results["shearlab FashionMNIST"][1], results["shearlab FashionMNIST"][1]), label="ave")
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
