module ScatteringTransform
using Distributed, SharedArrays
using LinearAlgebra, Shearlab, Interpolations, Wavelets, FFTW
using SpecialFunctions, LinearAlgebra
using HDF5, Plots, JLD



include("subsampling.jl")
export resample, sizes, bsplineType, bilinearType, autocorrType
include("modifiedTransforms.jl")
export WT, wavelet, cwt, getScales, computeWavelets
include("basicTypes.jl")
export layeredTransform, scattered
include("Utils.jl")
export getResizingRates, calculateThinStSizes, getPadBy, pad, outputSize, createFFTPlans, remoteMultiply, createRemoteFFTPlan
include("nonlinearities.jl")
export nonlinearity, absType, ReLUType, tanhType, softplusType, piecewiseType, spInverse, aTanh, Tanh, ReLU, piecewiseLinear, plInverse
# TODO: integrate path methods
include("pathMethods.jl")
export pathType, pathToThinIndex
include("transform.jl")
export st, transformMidLayer!, transformFinalLayer!,
    comparePathsChildren
include("inversion.jl")
export pseudoInversion
include("plotting.jl")
export flatten, MatrixAggrigator, plotCoordinate, reshapeFlattened,
    numberSkipped, logabs, maxPooling, numScales, incrementKeeper,
    numInLayer
include("postProcessing.jl")
export logabs, ReLU, MatrixAggrigator, reshapeFlattened,
    loadSyntheticMatFile, transformFolder, flatten

# TODO make a way to access a scattered2D by the index of (depth, scale,shearingFactor)
end
