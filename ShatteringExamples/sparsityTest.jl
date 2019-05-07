# using ScatteringTransform
using HDF5
using Plots, Statistics, PowerLaws


reorderCoeff(A) = sort(abs.(A), dims = length(size(A)),rev=true)

"""
    conPowerFit, KSstatistic = testSingleRow(A, i)
given an HDF5 pointer A and an example index i, compute estimate_xmin on the single row i
"""
function testSingleRow(A, i)
    return estimate_xmin(reshape(reorderCoeff(A[i,:]),(length(A[i,:]))), con_powerlaw)
end

"""
     (conPowers, KStestPValues, αs, θs) = testWholematrix(A)

given an HDF5 pointer A, compute estimate_xmin on every row separately
"""
function testWholeMatrix(A)
    αs = zeros(size(A,1))
    θs = zeros(size(A,1))
    conPowers = Array{con_powerlaw}(undef, size(A,1))
    for i = 1:size(A, 1)
        conPowers[i], KStestPValues[i] = testSingleRow(A, i)
        αs[i] = conPowers.α
        θs[i] = conPowers.θ
    end
    return (conPowers, KStestPValues, αs, θs)
end

"""
    coeffs, stderrs = fitPolydecay(A)

given a sorted, positive matrix, fit the rows with least squares on a log scale.
"""
function fitPolyDecay(A)
    coeffs = zeros(size(A,1), 2)
    stderrs = zeros(size(A,1), 2)
    for i=1:size(A,1)
        df = DataFrame(X=log.(reorderA[i,:]), Y=log.(1:size(A,2)))
        ols = lm(@formula(Y~X), df)
        coeffs[i,:] = coef(ols)
        stderrs[i,:] = stderror(ols)
    end 
    return (coeffs, stderrs)
end

function fitLine(coeffs, A)
    x = log.(1:size(A, 2))
    netLines = coeffs[:,2] * x' .+ coeffs[:,1]
    meanLine = mean(netLines, dims=1)
    stdLine = std(netLines, dims=1)
    return (meanLine, stdLine)
end



# load the dataset; row examples (nExamples×dims)
filename = "/fasterHome/workingDataDir/shattering/shatteredMNISTsoftplus2.h5"
Atmp = h5open(filename,"r")
A = Atmp["data/shattered"]

reorderA = reorderCoeff(A)
αs = zeros(size(A,1))
θs = zeros(size(A,1))
conPowers = Array{con_powerlaw}(undef, size(A,1))
KStestPValues = zeros(size(A,1))
for i = 1:size(A,1)
    conPowers[i], KStestPValues[i] = estimate_xmin(reorderA[i,:], con_powerlaw)
end
coeffs, stderrs = fitPolyDecay(reorderA)
println(coeffs)
plot(reorderA)


mean(coeffs[])
