__precompile__(false)
module ScatteringTransform
using Distributed, SharedArrays
using LinearAlgebra, Shearlab, Interpolations, Wavelets, FFTW
using SpecialFunctions, LinearAlgebra
using HDF5, Plots, JLD



include("subsampling.jl")
export resample, sizes, bsplineType, bilinearType, autocorrType
include("modifiedTransforms.jl")
export CFWA, WT, wavelet, cwt, getScales, computeWavelets
include("basicTypes.jl")
export layeredTransform, scattered
include("Utils.jl")
export getResizingRates, calculateThinStSizes, getPadBy,  outputSize
include("nonlinearities.jl")
export absType, ReLUType, tanhType, softplusType, spInverse, aTanh,
    Tanh, ReLU
# TODO: integrate path methods
include("pathMethods.jl")
export pathType, pathToThinIndex
include("transform.jl")
export st, transformMidLayer!, transformFinalLayer!,
    comparePathsChildren
include("plotting.jl")
export flatten, MatrixAggrigator, plotCoordinate, reshapeFlattened,
    numberSkipped, logabs, maxPooling, numScales, incrementKeeper,
    numInLayer
include("postProcessing.jl")
export logabs, ReLU, MatrixAggrigator, reshapeFlattened,
    loadSyntheticMatFile, transformFolder

# TODO make a way to access a scattered2D by the index of (depth, scale,shearingFactor)
end
