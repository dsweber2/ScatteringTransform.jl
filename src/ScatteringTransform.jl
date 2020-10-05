module ScatteringTransform

using ChainRules
using CUDA
using Flux:mse, update!
using FFTW
using Zygote, Flux, Shearlab, LinearAlgebra, AbstractFFTs
using Flux
using FourierFilterFlux
using Adapt
using RecipesBase
using Base: tail
using ChainRulesCore
using Plots # who cares about weight really?
using Statistics, LineSearches, Optim

import Adapt: adapt
import ChainRules:rrule
import Zygote:has_chain_rrule, rrule


abstract type scatteringTransform{Dimension, Depth} end
struct stFlux{Dimension, Depth, ChainType,D,E,F} <: scatteringTransform{Dimension, Depth}
    mainChain::ChainType
    normalize::Bool
    outputSizes::D
    outputPool::E
    settings::F
end

import Base.ndims
ndims(s::scatteringTransform{D}) where D = D
nPathDims(ii) = 1+max(min(ii-2,1),0) # the number of path dimensions at layer ii (zeroth
# is ii=1)
function Base.show(io::IO, st::stFlux{Dim,Dep}) where {Dim,Dep}
    layers = st.mainChain.layers
    σ = layers[1].σ
    Nd = ndims(st)
    nFilters = [size(layers[i].weight,3)-1 for i=1:3:(3*Dim)]
    batchSize = getBatchSize(layers[1])
    print(io, "stFlux{$(Dep), Nd=$(Nd), filters=$(nFilters), σ = " *
          "$(σ), batchSize = $(batchSize), normalize = $(st.normalize)}")
end

# the type T is a type of frame transform that forms the backbone of the transform
# the type Dimension<:Integer gives the dimension of the transform

struct stParallel{T, Dimension, Depth} <: scatteringTransform{Dimension, Depth}
    n::Tuple{Vararg{Int, Dimension}} # the dimensions of a single entry
    shears::Array{T,1} # the array of the transforms; the final of these is
    # used only for averaging, so it has length m+1
    subsampling::Array{Float32, 1} # for each layer, the rate of
    # subsampling. There is one of these for layer zero as well, since the
    # output is subsampled, so it should have length m+1
    outputSize::Array{Int, 2} # a list of the size of a single output example
    # dimensions in each layer. The first index is layer, the second is
    # dimension (e.g. a 3 layer shattering transform is 3×2) TODO: currently
    # unused for the 1D case
end

function Base.show(io::IO, l::stParallel{T,D,Depth}) where {T,D,Depth}
    print(io, "stParallel{$T,$D} depth $(Depth), input size $(l.n), subsampling rates $(l.subsampling), outputsizes = $(l.outputSize)")
end

function eltypes(f::stParallel)
    return (eltype(f.shears), length(f.n))
end

export scatteringTransform, stFlux, stParallel, eltypes
include("shared.jl")
include("pathLocs.jl")
export pathLocs 
include("scattered.jl")
export Scattered, ScatteredFull, ScatteredOut, pathLocs, nonZeroPaths
include("parallel/basicTypes.jl")
include("Flux/flPool.jl")
export RationPool, nPoolDims, outdims, poolSize
include("Flux/flTransform.jl")
export normalize

function scatteringTransform(inputSize, m, backend::Val{stFlux}; kwargs...)
    stFlux(inputSize, m; kwargs...)
end

function scatteringTransform(inputSize, m, backend::Val{stParallel}; kwargs...)
    stParallel(m, inputSize[1]; kwargs...)
end
scatteringTransform(inputSize,m; kwargs...) = 
    scatteringTransform(inputSize, m, backend=stFlux; kwargs...)

include("Flux/flUtilities.jl")
export getWavelets, flatten, roll, importantCoords, batchOff
include("parallel/parallelCore.jl") # there's enough weird stuff going on in
# here that I'm isolating it in a module
function (stPara::stParallel)(x; nonlinearity=abs, thin=false,
                              outputSubsample=(-1,-1), subsam=true,
                              totalScales=[-1 for i=1:layers.m+1],
                              percentage=.9, fftPlans=-1)
   st(x, stPara, nonlinearity; thin=thin, outputSubsample=outputSubsample,
      subsam=subsam, totalScales=totalScales, percentage=percentage,
      fftPlans=fftPlans)
end

roll(toRoll, st::stParallel; varargs...) = wrap(st,toRoll; varargs...)
export roll, wrap, sizes, calculateThinStSizes, createFFTPlans,
    createRemoteFFTPlan, computeAllWavelets, plotAllWavelets
export spInverse, aTanh, Tanh, ReLU, piecewiseLinear, plInverse
export st
export pathToThinIndex, MatrixAggregator, plotCoordinate, reshapeFlattened,
    numberSkipped, maxPooling, numScales, incrementKeeper, numInLayer
export loadSyntheticMatFile, transformFolder, flatten
include("Flux/adjoints.jl")
include("Flux/interpretationTools.jl")
export ∇st, plotFirstLayer1D, gifFirstLayer, plotSecondLayer1D, addNextPath
include("Flux/trainingStorageTools.jl")
export buildRecord, expandRecord!, addCurrent!, makeObjFun, fitReverseSt,
    justTrain, maximizeSingleCoordinate, fitByPerturbing, perturb, genNoise,
    chooseLargest, continueTrain, fitUsingOptim
end # end Module
