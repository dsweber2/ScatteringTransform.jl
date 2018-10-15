# this file is best used interactively in atom
using Pkg; Pkg.activate("/home/dsweber/.julia/dev/ScatteringTransform.jl")
using ScatteringTransform, Plots
# simple example; a sum of sine waves that has a discontinuous jump
t = 0:6π/100:6π
f = (sin.(t) + 1/3*sin.(π*t+π/3)) + max.(0,t-3*π)./(t-3*π)
plot(t, f)

# default transform uses Morlet wavelets
layeredTransform(3,f)
