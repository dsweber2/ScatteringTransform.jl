# using ScatteringTransform 
using HDF5
using Plots, Statistics, PowerLaws


reorderCoeff(A::Array{T,2}) where T = sort(abs.(A), dims = length(size(A)),rev=true)
reorderCoeff(A::Array{T,1}) where T = sort(abs.(A), rev=true)
"""
    conPowerFit, KSstatistic = testSingleRow(A, i)
given an HDF5 pointer A and an example index i, compute estimate_xmin on the single row i
"""
function testSingleRow(A, i)
    thing = reorderCoeff(A[i, :])
    if length(thing)>1
        return estimate_xmin(reshape(reorderCoeff(A[i,:]), (length(A[i,:],))), con_powerlaw)
end

"""
     (conPowers, KStestPValues, αs, θs) = testWholematrix(A)

given an HDF5 pointer A, compute estimate_xmin on every row separately
"""
function testWholeMatrix(A)
    αs = zeros(size(A,1))
    θs = zeros(size(A,1))
    conPowers = Array{con_powerlaw}(undef, size(A,1))
    KStestPValues = zeros(size(A,1))
    for i = 1:size(A, 1)
        conPowers[i], KStestPValues[i] = testSingleRow(A, i)
        αs[i] = conPowers[i].α
        θs[i] = conPowers[i].θ
    end
    return (conPowers, KStestPValues, αs, θs)
end
testSingleRow(zeros(10,10),1)
# load the dataset; row examples (nExamples×dims)
filename = "/fasterHome/workingDataDir/shattering/shatteredMNISTsoftplus2.h5"
Atmp = h5open(filename,"r")
A = Atmp["data/shattered"]
conPowers, KStest, αs, θs = testWholeMatrix(A)

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
