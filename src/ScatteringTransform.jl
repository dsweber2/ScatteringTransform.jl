module ScatteringTransform

using ChainRules
using CUDA
using FFTW
using Wavelets, ContinuousWavelets
using Zygote, Flux, LinearAlgebra, AbstractFFTs
using Flux
using FourierFilterFlux
using Adapt
using RecipesBase
using Base: tail
using ChainRulesCore
using Plots # who cares about weight really?
using Statistics
using Dates

import Adapt: adapt
import ChainRules: rrule
import Zygote: has_chain_rrule, rrule
import Wavelets: eltypes


include("shared.jl")
export scatteringTransform, stFlux, stParallel, eltypes, depth
include("pathLocs.jl")
export pathLocs
include("scattered.jl")
export Scattered, ScatteredFull, ScatteredOut, nonZeroPaths
include("pool.jl")
export RationPool, nPoolDims, outdims, poolSize
include("transform.jl")
export normalize, cu

function scatteringTransform(inputSize, m, backend::UnionAll; kwargs...)
    backend(inputSize, m; kwargs...)
end

scatteringTransform(inputSize, m; kwargs...) =
    scatteringTransform(inputSize, m, stFlux; kwargs...)

include("utilities.jl")
export getWavelets, flatten, roll, importantCoords, batchOff, getParameters, getMeanFreq
export roll, wrap, flatten
include("adjoints.jl")
include("interpretationTools.jl")
export ∇st, plotFirstLayer1D, gifFirstLayer, plotSecondLayer, addNextPath, jointPlot
end # end Module
