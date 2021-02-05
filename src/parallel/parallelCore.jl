module parallel
using Distributed, SharedArrays
using LinearAlgebra, Interpolations, Wavelets, FFTW
using ContinuousWavelets
using SpecialFunctions, LinearAlgebra
using HDF5, JLD
using Plots
# todo remove nScales controls
# the overall container defines a few types we need here
import ScatteringTransform:stParallel, Scattered, ScatteredFull, ScatteredOut, depth, eltypes

include("subsampling.jl")
export sizes, bsplineType, bilinearType
include("modifiedTransforms.jl")
include("Utils.jl")
export calcuateSizes, calculateThinStSizes, createFFTPlans, remoteMultiply,
    createRemoteFFTPlan, computeAllWavelets, plotAllWavelets, getQ
include("nonlinearities.jl")
export spInverse, aTanh, Tanh, ReLU, piecewiseLinear, plInverse
include("transform.jl")
export st, transformMidLayer!, transformFinalLayer!
include("pathMethods.jl")
export pathToThinIndex, MatrixAggrigator, plotCoordinate,
    reshapeFlattened, numberSkipped, logabs, maxPooling, numScales,
    incrementKeeper, numInLayer
include("postProcessing.jl")
export logabs, ReLU, MatrixAggrigator, reshapeFlattened,
    loadSyntheticMatFile, transformFolder, flatten, roll
end
