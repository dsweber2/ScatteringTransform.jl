module ScatteringTransform

using Core: @__doc__
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
export Scattered, ScatteredFull, ScatteredOut, nonZeroPaths, cat, addNextPath
include("pool.jl")
export RationPool, nPoolDims, outdims, poolSize
include("transform.jl")
export cu

@doc """
     scatteringTransform(inputSize, m=2, backend::UnionAll=stFlux; kwargs...)
 The constructor for the abstract type scatteringTransform, with the concrete type specified by `backend`.
 """
function scatteringTransform(inputSize, m=2, backend::UnionAll=stFlux; kwargs...)
    backend(inputSize, m; kwargs...)
end

include("utilities.jl")
export getWavelets, flatten, roll, importantCoords, batchOff, getParameters, getMeanFreq, computeLoc
export roll, wrap, flatten
include("adjoints.jl")
end # end Module
