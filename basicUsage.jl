
# this file is best used interactively in atom
using Distributed
addprocs(4)
@everywhere using Revise
@everywhere using ScatteringTransform
using Plots
# simple example; a sum of sine waves that has a discontinuous jump
t = 0.001:6π/100:6π
f = (sin.(10*t) + 1/3*sin.(π*t.+π/3)) .+ 300 .* max.(0,t.-3*π)./(t.-3*π)
plot(t, f)

# brief example of what the wavelets look like


# First create the transform configuration
layers = layeredTransform(2,f)

# two major versions:
###########################################################################################################
#####    the first outputs a datatype specifically for working with Scattering transforms, and has both all internal results and the output. It is not particulary appropriate for use on large datasets
###########################################################################################################

@time output = st(f,layers,nonlinear=abs)
firstLayer = zeros(25, 54)
for i=1:54
  firstLayer[:,i] = bspline(bspline(abs.(cwt(f)[i,:]), 2),2)
end
heatmap(firstLayer')
tmp = cwt(f,layers.shears[1])
thing = zeros(Complex{Float64}, 50, 33)
cwt(bspline(tmp[1,:,1], 2),layers.shears[2])
for i=1:33
  thing[:,i]= cwt(bspline(abs.(tmp[1,:,i+1]), 2),layers.shears[2])[1,:,1]
end
thing
heatmap(abs.(thing)')
heatmap(abs.(cwt(reshape(f,(1,size(f)...)),layers.shears[1])[1,:,:])')
heatmap(abs.(output.data[2][1,:,:])')
heatmap(output.output[2][1,:,:]')
output.output[2]

# let's take a look at some of the coefficients
# first, for comparison, the cwt using the same Morlet wavelets is
heatmap(abs.(cwt(f,layers.shears[1])[1,:,:])', xaxis="Space",yaxis="Frequency")
# where the discontinuity is clearly visible
# For comparison, the first layer (note the difference in magnitude)
heatmap(output.output[2][1,:,:]', xlabel="space", ylabel="Frequency",title="First Layer Output")
# specifically, the highest scale/lowest frequency in the first layer, denoted by pathType(1,[1])
comparePathsChildren(output, pathType(1,[1]), layers)
# the other end
comparePathsChildren(output,pathType(1,[33]),layers)
# looking back at line 44, the 16th frequency looks like it should have some interesting behaviour
comparePathsChildren(output,pathType(1,[33]),layers)

###########################################################################################################
####    the second outputs a single array where the last dimension corresponds to the transformed data. Useful for large datasets, and more optimized
###########################################################################################################

@time output = thinSt(f, layers, outputSubsample=(-1,3))
# output subsample as written only takes 3 values from each output path

# plot the actual wavelets used
daughters = computeWavelets(f, layers.shears[1])
plotlyjs()
daughters
plot(reverse(abs.(daughters)[1:100,:],dims=2), title="Absolute Value of the Morlet Wavelets", label="")
heatmap(abs.(daughters[1:100,:]'), title="Absolute Value of the Morlet Wavelets")
savefig("/home/dsweber/allHail/SonarProjectSaito/plots/SSAM2Plots/MorletWavelets.pdf")
