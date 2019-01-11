# this file is best used interactively in atom
using Pkg; Pkg.activate("/home/dsweber/.julia/dev/ScatteringTransform.jl")
using Revise
using Distributed
addprocs(9)
include("src/ScatteringTransform.jl")
using ScatteringTransform, Plots
# simple example; a sum of sine waves that has a discontinuous jump
t = 0:6π/100:6π
f = (sin.(t) + 1/3*sin.(π*t+π/3)) + max.(0,t-3*π)./(t-3*π)
plot(t, f)

# default transform uses Morlet wavelets
layeredTransform(3,f)

# tmpcode
lengthOfALayer
sum(lengthOfALayer[1:(2)])
k, n, q, dataSizes, outputSizes, resultingSize = ScatteringTransform.calculateThinStSizes(layers, outputSubsample, originalDimensions)
thinStDims = pathToThinIndex(path, layers, outputSubsample)
toPlot = thinResult[axes(thinResult)[1:end-1]..., thinStDims]
toPlot = reshape(toPlot,(size(thinResult)[1:end-1]...,floor(Int,length(thinStDims)/resultingSize[path.m+1]), resultingSize[path.m+1]))
plotlyjs()
ranges = (minimum(toPlot),maximum(toPlot))
colorMeMine = cgrad(:balance,scale=scale)
plot([heatmap(toPlot[i,:,:], xticks=1:size(toPlot)[end], clims=ranges, fillcolor=colorMeMine) for i=1:size(toPlot,1)]...)
k, n, q, dataSizes, outputSizes, resultingSize = ScatteringTransform.calculateThinStSizes(layers, (-1,3), (20032,420))
lengthOfALayer = resultingSize.*q
initIndex = sum(lengthOfALayer[1:(path.m)])
tmpNlayer = [[numScales(layers.shears[j], layers.n)-1 for j=1:path.m]; 1]
[prod(tmpNlayer[i+1:path.m]) for i=1:path.m]
path.Idxs'*[prod(tmpNlayer[i+1:path.m]) for i=1:path.m]



# A version that only has the subsampled output, primarily useful for classification
layers = layeredTransform(2,420)
f = randn(2000,30,420)
outt = thinSt(f,layers)
reshape(1:24,(3,8))


# plot the actual wavelets used
layers = layeredTransform(2,420)
f = randn(2,420)

daughters = computeWavelets(f, layers.shears[1])
plotlyjs()
daughters
plot(reverse(abs.(daughters)[1:150,:],dims=2),title="Absolute Value of the Morlet Wavelets used")
normalizeddaughters = daughters./sqrt.(sum(abs.(daughters).^2,dims=1))
plot(reverse(abs.(normalizeddaughters)[1:150,:],dims=2),title="Absolute Value of the Morlet Wavelets used")
savefig("/home/dsweber/allHail/SonarProjectSaito/plots/SSAM2Plots/MorletWavelets.pdf")
reverse(abs.(daughters),dims=1)
tmp = randn(10,5)
sum(abs.(tmp./sum(abs.(tmp).^2,dims=1)).^2,dims=1)
tmp = [1 4; 2 5; 3 6]
./[1 2 3]'
sum((tmp./sqrt.(sum((tmp).^2,dims=1))).^2,dims=1)
