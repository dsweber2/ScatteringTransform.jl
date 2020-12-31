module ScatteringTransform

using ChainRules
using CUDA
using Flux:mse, update!
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
import ChainRules:rrule
import Zygote:has_chain_rrule, rrule
import Wavelets:eltypes


include("shared.jl")
export scatteringTransform, stFlux, stParallel, eltypes, depth
include("pathLocs.jl")
export pathLocs 
include("scattered.jl")
export Scattered, ScatteredFull, ScatteredOut, pathLocs, nonZeroPaths
include("parallel/basicTypes.jl")
include("Flux/flPool.jl")
export RationPool, nPoolDims, outdims, poolSize
include("Flux/flTransform.jl")
export normalize

function scatteringTransform(inputSize, m, backend::UnionAll; kwargs...)
    backend(inputSize, m; kwargs...)
end

scatteringTransform(inputSize,m; kwargs...) =
    scatteringTransform(inputSize, m, stFlux; kwargs...)

include("Flux/flUtilities.jl")
export getWavelets, flatten, roll, importantCoords, batchOff
include("parallel/parallelCore.jl") # there's enough weird stuff going on in
# here that I'm isolating it in a module
using .parallel
function (stPara::stParallel)(x; nonlinearity=abs, thin=false,
                              outputSubsample=(-1,-1), subsam=true,
                              totalScales=[-1 for i=1:depth(stPara)+1],
                              percentage=.9, fftPlans=-1)
   st(x, stPara, nonlinearity; thin=thin, outputSubsample=outputSubsample,
      subsam=subsam, totalScales=totalScales, percentage=percentage,
      fftPlans=fftPlans)
end

function ScatteredFull(layers::scatteringTransform{S,1}, X::Array{T,N};
                       totalScales=[-1 for i=1:depth(layers) +1],
                       outputSubsample=(-1,-1)) where {T <: Real, N, S}
    if N == 1
        X = reshape(X, (size(X)..., 1));
    end

    n, q, dataSizes, outputSizes, resultingSize = 
        parallel.calculateSizes(layers, outputSubsample, size(X), 
                                totalScales = totalScales)
    numInLayer = getQ(layers, n, totalScales; product=false)
    addedLayers = [numInLayer[min(i,2):i] for i=1:depth(layers) + 1]
    if 1==N
        zerr = [zeros(T, n[i], addedLayers[i]...) for i=1:depth(layers)+1]
        output = [zeros(T, resultingSize[i], addedLayers[i]...) for i=1:depth(layers)+1]
    else
        zerr=[zeros(T, n[i], q[i], size(X)[2:end]...) for
              i=1:depth(layers)+1]
        output = [zeros(T, resultingSize[i], addedLayers[i]..., size(X)[2:end]...)
                  for i=1:depth(layers)+1]
        @info "" [size(x) for x in output]
    end
    zerr[1][:, 1, axes(X)[2:end]...] = copy(X)
    return ScatteredFull{T, N+1}(depth(layers), 1, zerr, output)
end

function ScatteredOut(layers::ST, X::Array{T,N};
                       totalScales=[-1 for i=1:depth(layers) +1],
                       outputSubsample=(-1,-1)) where {ST <: scatteringTransform, T <: Real, N, S}
    if N == 1
        X = reshape(X, (size(X)..., 1));
    end

    n, q, dataSizes, outputSizes, resultingSize = 
        parallel.calculateSizes(layers, outputSubsample, size(X), totalScales =
                       totalScales)
    addedLayers = parallel.getListOfScaleDims(layers, n, totalScales)
    @info addedLayers
    if 1==N
        output = [zeros(T, resultingSize[i], addedLayers[i]...) for i=1:depth(layers)+1]
    else
        output = [zeros(T, resultingSize[i], addedLayers[i]..., size(X)[2:end]...)
                  for i=1:depth(layers)+1]
    end
    @info [size(x) for x in output]
    return ScatteredOut{T, N+1}(depth(layers), 1, output)
end

roll(toRoll, stP::stParallel, originalInput; varargs...) = parallel.roll(stP, toRoll,originalInput; varargs...)
export roll, wrap, sizes, calculateThinStSizes, createFFTPlans,
    createRemoteFFTPlan, computeAllWavelets, plotAllWavelets
export spInverse, aTanh, Tanh, ReLU, piecewiseLinear, plInverse
export st
export pathToThinIndex, MatrixAggregator, plotCoordinate, reshapeFlattened,
    numberSkipped, maxPooling, numScales, incrementKeeper, numInLayer
export loadSyntheticMatFile, transformFolder, flatten
include("Flux/adjoints.jl")
include("Flux/interpretationTools.jl")
export ∇st, plotFirstLayer1D, gifFirstLayer, plotSecondLayer, addNextPath
include("Flux/trainingStorageTools.jl")
export buildRecord, expandRecord!, addCurrent!, makeObjFun, fitReverseSt,
    justTrain, maximizeSingleCoordinate, fitByPerturbing, perturb, genNoise,
    chooseLargest, continueTrain, fitUsingOptim
end # end Module
