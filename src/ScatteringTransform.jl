module ScatteringTransform

using ChainRules
using CUDA
using Flux: mse, update!
using FFTW
using Wavelets, ContinuousWavelets
using Zygote, Flux, Shearlab, LinearAlgebra, AbstractFFTs
using Flux
using FourierFilterFlux
using Adapt
using RecipesBase
using Base: tail
using ChainRulesCore
using Plots # who cares about weight really?
using Statistics, LineSearches, Optim
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
export Scattered, ScatteredFull, ScatteredOut, pathLocs, nonZeroPaths
include("Flux/flPool.jl")
export RationPool, nPoolDims, outdims, poolSize
include("Flux/flTransform.jl")
export normalize
export cu

function scatteringTransform(inputSize, m, backend::UnionAll; kwargs...)
    backend(inputSize, m; kwargs...)
end

scatteringTransform(inputSize, m; kwargs...) =
    scatteringTransform(inputSize, m, stFlux; kwargs...)

include("Flux/flUtilities.jl")
export getWavelets, flatten, roll, importantCoords, batchOff, getParameters, getMeanFreq


roll(toRoll, stP::stParallel, originalInput; varargs...) = parallel.roll(stP, toRoll, originalInput; varargs...)
export roll, wrap, sizes, calculateThinStSizes, createFFTPlans,
    createRemoteFFTPlan, computeAllWavelets, plotAllWavelets
export spInverse, aTanh, Tanh, ReLU, piecewiseLinear, plInverse
export st
export pathToThinIndex, MatrixAggregator, plotCoordinate, reshapeFlattened,
    numberSkipped, maxPooling, numScales, incrementKeeper, numInLayer
export loadSyntheticMatFile, transformFolder, flatten
include("Flux/adjoints.jl")
include("Flux/interpretationTools.jl")
export ∇st, plotFirstLayer1D, gifFirstLayer, plotSecondLayer, addNextPath, jointPlot
end # end Module
