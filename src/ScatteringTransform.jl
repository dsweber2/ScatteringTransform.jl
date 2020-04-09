module ScatteringTransform
using Distributed, SharedArrays
using LinearAlgebra, Interpolations, Wavelets, FFTW
using SpecialFunctions, LinearAlgebra
using HDF5, JLD
using Plots
# todo remove nScales controls

# TODO: integrate path methods
struct pathType{T}
  m::Int64
  Idxs::Array{T,1}
  function pathType(m::Int64,Idxs)
      @assert m==length(Idxs)
      @assert minimum([true,[tp <: Integer for tp in typeof.(Idxs[1:end-1])]...])
      new{eltype(Idxs)}(m, Idxs)
  end
end

"""
    path = pathType(m::Int64, indexInShear::Array{Int64,1}, layers::layeredTransform)

path constructor that uses the fixed index in each layer (i.e. the index of the shearlet filter in shearletIdxs)
"""
function pathType(indexInLayers::Array)
  pathType(length(indexInLayers), indexInLayers)
end

import Base:show
function Base.show(io::IO, p::pathType{T}) where T
    pl = p.Idxs[end]
    if typeof(pl)<:Colon
        print(io, "pathType[$(p.Idxs[1:end-1]...), :]")
    else
        print(io, "pathType$(p.Idxs)")
    end
end



export pathType
include("subsampling.jl")
export resample, sizes, bsplineType, bilinearType, autocorrType
include("modifiedTransforms.jl")
export WT, wavelet, cwt, getScales, computeWavelets
include("basicTypes.jl")
export layeredTransform, scattered
include("Utils.jl")
export calculateThinStSizes, getPadBy, pad, outputSize, createFFTPlans, remoteMultiply, createRemoteFFTPlan, computeAllWavelets, plotAllWavelets
include("nonlinearities.jl")
export nonlinearity, absType, ReLUType, tanhType, softplusType, piecewiseType, spInverse, aTanh, Tanh, ReLU, piecewiseLinear, plInverse
include("transform.jl")
export st, transformMidLayer!, transformFinalLayer!
include("pathMethods.jl")
export pathToThinIndex
export flatten, MatrixAggrigator, plotCoordinate, reshapeFlattened,
    numberSkipped, logabs, maxPooling, numScales, incrementKeeper,
    numInLayer
include("postProcessing.jl")
export logabs, ReLU, MatrixAggrigator, reshapeFlattened,
    loadSyntheticMatFile, transformFolder, flatten

# TODO make a way to access a scattered2D by the index of (depth, scale,shearingFactor)
end
