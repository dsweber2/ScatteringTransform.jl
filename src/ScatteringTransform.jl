module ScatteringTransform

using ChainRules
using CUDA
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
export scatteringTransform, stFlux, depth
include("pathLocs.jl")
export pathLocs
include("scattered.jl")
export Scattered, ScatteredFull, ScatteredOut, nonZeroPaths, cat
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
end # end Module
