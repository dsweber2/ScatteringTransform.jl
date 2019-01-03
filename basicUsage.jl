# this file is best used interactively in atom
using Pkg; Pkg.activate("/home/dsweber/.julia/dev/ScatteringTransform.jl")
using ScatteringTransform, Plots
# simple example; a sum of sine waves that has a discontinuous jump
t = 0:6π/100:6π
f = (sin.(t) + 1/3*sin.(π*t+π/3)) + max.(0,t-3*π)./(t-3*π)
plot(t, f)

# default transform uses Morlet wavelets
layeredTransform(3,f)






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
