
# this file is best used interactively in atom
using Distributed
addprocs(4)
@everywhere using Revise
@everywhere using ScatteringTransform
using Plots
using Statistics
# simple example; a sum of sine waves that has a discontinuous jump
t = 0.001:6π/100:6π
f = 2 .* max.(0,-(t.-3*π))./(t.-3*π) .* (sin.(2π*t)) .+ 10 .* max.(0,t.-3*π)./(t.-3*π) .+ max.(0,t.-3*π)./(t.-3*π).* sin.(4π*t.+π)
plot(t, f)

# brief example of what the wavelets look like


# First create the transform configuration
layers = layeredTransform(2, length(f))

# two major versions:
###########################################################################################################
#####    the first outputs a datatype specifically for working with Scattering transforms, and has both all internal results and the output. It is not particulary appropriate for use on large datasets
###########################################################################################################

@time output = st(f, layers, absType(), thin=false) # run twice to get the time. The first is compile time

###########################################################################################################
####    the second outputs a single array where the last dimension corresponds to the transformed data. Useful for large datasets, and more optimized
###########################################################################################################
@time thinOp = st(f, layers, absType(), outputSubsample=(-1,3), thin=true) # run twice to get the time. The first is compile time


# computing the morlet wavelet coefficients subsampled for comparison
waves = wavelet(WT.Morlet(5.0), averagingType = WT.Father())
cwtf = cwt(f, waves)
firstLayer = zeros(eltype(cwtf), 25, size(cwtf, 2))
for i=1:size(firstLayer,2)
  firstLayer[:,i] = ScatteringTransform.bspline(ScatteringTransform.bspline(abs.(cwt(f, waves)[:,i]), 2), 2)
end

plot(abs.(firstLayer)[:,end,1])
# let's take a look at some of the coefficients
# first, for comparison, the cwt using the same Morlet wavelets is
heatmap(abs.(firstLayer)[:,2:end,1]', xaxis="Space",yaxis="Frequency", title="Morlet coefficients of f")
# where the discontinuity is clearly visible, as is the difference in frequency
# For comparison, the first layer (the difference in magnitude is due to normalization)
heatmap(output[2][:,:,1]', xlabel="space", ylabel="Frequency",title="First Layer Output (smoothed version of the previous)")
# the discontinuity is less visible here, but it remains in the next layer

# let's take a closer look at the 6th frequency, which is where the sinusoid above is represented
p = pathType([6]) # an object that we can use as an index into the output
plot(output[p], title="6th frequency coefficients") # biased towards the left, where the oscillation is present
# now it's children
p = pathType([6,:])
heatmap(output[p][:,:,1]', xlabel="space", ylabel="Frequency", title="6th frequency variation")
# now the highest frequency, which should reflect the discontinuity
p = pathType([15,:])
heatmap(output[p][:,:,1]', xlabel="space", ylabel="Frequency", title="15th frequency decomposition")
p = pathType([14,:])
heatmap(output[p][:,:,1]', xlabel="space", ylabel="Frequency", title="15th frequency decomposition")





# An example from a sonar dataset
using JLD2
@load "examples/resp1999.0_10.2.jld2"
resp = permutedims(resp, (1,3,2))
# this example is the reflection off of a sharkfin whose interior liquid has speed 1999.0 at distance 10.2 meters
# it is now ordered as (range)×(cross-range)×(angle)

layers = layeredTransform(2, size(resp,1))
size(resp)
@time output = st(resp, layers, absType()) # run twice to get the time. The first is compile time
