# how should we apply a function recursively to the stFlux type? to each part of the chain, of course
function mapEvery3(to, ii, x)
    if ii % 3 == 1
        adapt(to, x)
    else
        x
    end
end
import FourierFilterFlux.cu
function cu(stf::stFlux{Dimension,Depth,ChainType,D,E,F}) where {Dimension,Depth,ChainType,D,E,F}
    newChain = Chain((map(iix -> (iix[1] % 3 == 1 ? cu(iix[2]) : iix[2]), enumerate(stf.mainChain)))...)
    return stFlux{Dimension,Depth,typeof(newChain),D,E,F}(newChain, stf.normalize, stf.outputSizes, stf.outputPool, stf.settings)
end
import Adapt.adapt
function adapt(to, stf::stFlux{Dimension,Depth,ChainType,D,E,F}) where {Dimension,Depth,ChainType,D,E,F}
    newChain = Chain((map(iix -> mapEvery3(to, iix...), enumerate(stf.mainChain)))...)
    return stFlux{Dimension,Depth,typeof(newChain),D,E,F}(newChain, stf.normalize, stf.outputSizes, stf.outputPool, stf.settings)
end

function adapt(to, stResult::ScatteredFull{T,N}) where {T,N}
    data = adapt(to, stResult.data)
    output = adapt(to, stResult.output)
    return ScatteredFull{eltype(data),N}(stResult.m, stResult.k, data, output)
end

function adapt(to, stResult::ScatteredOut{T,N}) where {T,N}
    output = adapt(to, stResult.output)
    return ScatteredOut{eltype(output),N}(stResult.m, stResult.k, output)
end
export adapt

"""
    getWavelets(sc::stFlux; spaceDomain=false) -> wave1, wave2, wave3, ...

Get the wavelets used in each layer. If `spaceDomain` is `true`, then it will also convert the filters from the stored positive Fourier representation to a space version.
"""
function getWavelets(sc::stFlux; spaceDomain=false)
    freqDomain = map(x -> x.weight, filter(x -> (typeof(x) <: ConvFFT), sc.mainChain.layers)) # filter to only have ConvFFTs, and then return the wavelets of those
    if spaceDomain
        return map(originalDomain, filter(x -> (typeof(x) <: ConvFFT), sc.mainChain.layers))
    else
        return map(x -> x.weight, filter(x -> (typeof(x) <: ConvFFT), sc.mainChain.layers))
    end
end

import ContinuousWavelets: getMeanFreq
"""
    getMeanFreq(sc::stFlux{1}, δt=1000)
Get a list of the mean frequencies for the filter bank in each layer. The averaging filter is last, and gives the mean frequency of the positive frequency only. Note that `δt` gives the sampling rate for the input only, and that it decreases at each subsequent layer at the rate implied by the subsampling in `sc`.
```jldoctest
julia> using ScatteringTransform

julia> St = scatteringTransform((1024,1,1),2)
stFlux{2, Nd=1, filters=[15], σ = abs, batchSize = 1, normalize = true}

julia> f1, f2, f3 = getMeanFreq(St);

julia> f1'
1×16 adjoint(::Vector{Float64}) with eltype Float64:
 7.70368  54.4302  78.7967  …  315.712  338.416  18.6697
julia> f2'
1×15 adjoint(::Vector{Float64}) with eltype Float64:
 10.8253  64.1205  89.7788  …  296.729  317.265  22.1889
```
"""
function getMeanFreq(sc::stFlux{1}, δt=1000)
    waves = getWavelets(sc)[1:end-1]
    shrinkage = [size(waves[i+1], 1) / size(waves[i], 1) for i = 1:length(waves)-1]
    δts = δt * [1, shrinkage...]
    freqs = map(getMeanFreq, waves, δts)
    return (freqs..., [zero(freqs[1][1])])
end

function getMeanFreq(Ŵ::Tuple, fsample=2000)
    eachNorm = [norm(w, 1) for w in Ŵ]
    freqs = range(0, fsample / 2, length=length(Ŵ[1]))
    return map(ŵ -> sum(abs.(ŵ) .* freqs), Ŵ) ./ eachNorm
end

"""
    flatten(scatRes) -> output
given `scatRes`, a scattered output or full, it produces a single vector containing the entire transform in order, i.e. the same format as output by thinSt.
"""
function flatten(scatRes::S) where {S<:Scattered}
    return scatRes[:]
end
flatten(scatRes) = scatRes


"""
    roll(toRoll, st::stFlux)
Given a scattering transform `st` and an array `toRoll` that is `NCoeffs×extraDims`, "roll" up `toRoll` into a `ScatteredOut`.
"""
function roll(toRoll, st::stFlux)
    Nd = ndims(st)
    oS = st.outputSizes
    roll(toRoll, oS, Nd)
end

function roll(toRoll, stOutput::S) where {S<:Scattered}
    Nd = ndims(stOutput)
    oS = map(size, stOutput.output)
    return roll(toRoll, oS, Nd)
end

function roll(toRoll, oS::Tuple, Nd)
    nExamples = size(toRoll)[2:end]
    rolled = ([adapt(typeof(toRoll), zeros(eltype(toRoll),
        sz[1:Nd+nPathDims(ii)]...,
        nExamples...)) for (ii, sz) in
               enumerate(oS)]...,)

    locSoFar = 0
    for (ii, x) in enumerate(rolled)
        szThisLayer = oS[ii][1:Nd+nPathDims(ii)]
        totalThisLayer = prod(szThisLayer)
        range = (locSoFar+1):(locSoFar+totalThisLayer)
        addresses = (szThisLayer..., nExamples...)
        rolled[ii][:] = reshape(toRoll[range, :], addresses)
        locSoFar += totalThisLayer
    end
    return ScatteredOut(rolled, Nd)
end


"""
    p = computeLoc(loc, toRoll, st::stFlux)
given a location `loc` in the flattened output `toRoll`, return a
pathLocs describing that location in the rolled version
"""
function computeLoc(loc, toRoll, st::stFlux)
    Nd = ndims(st)
    oS = st.outputSizes
    nExamples = size(toRoll)[2:end]
    locSoFar = 0
    for (ii, x) in enumerate(oS)
        szThisLayer = x[1:Nd+nPathDims(ii)]
        totalThisLayer = prod(szThisLayer)
        if locSoFar + totalThisLayer ≥ loc
            return pathLocs(ii - 1, Tuple(CartesianIndices(szThisLayer)[1+loc-locSoFar]))
        end
        locSoFar += totalThisLayer
    end
end

"""
    importantCoords(scatRes)
given a `ScatteredOut` `scatRes`, make a list that gives the largest value on each path.
"""
function importantCoords(scatRes)
    return [dropdims(maximum(abs.(x), dims=(1, 2)), dims=(1, 2)) for x in scatRes]
end



"""
    batchOff(stack, x, batchSize)

transform `x` using `stack`, but where `x` and `stack` may have different batch sizes (the final dimension).
"""
function batchOff(stack, x, batchSize)
    nRounds = ceil(Int, size(x)[end] // batchSize)
    firstRes = stack(x[:, :, :, 1:batchSize])
    result = cu(zeros(size(firstRes)[1:end-1]..., size(x)[end]))
    result[:, 1:batchSize] = firstRes
    for i = 2:(nRounds-1)
        result[:, 1+(i-1)*batchSize:(i*batchSize)] = stack(x[:, :, :, 1+(i-1)*batchSize:(i*batchSize)])
    end
    result[:, (1+(nRounds-1)*batchSize):end] = stack(cat(x[:, :, :,
            (1+(nRounds-1)*batchSize):end],
        cu(zeros(size(x)[1:3]...,
            nRounds * batchSize
            -
            size(x, 4))),
        dims=4))[:, 1:(size(x, 4)-(nRounds-1)*batchSize)]
    return result
end


"""
    getParameters(st, s)
given a `scatteringTransform` object and a symbol `s` representing a possible keyword, e.g. `:Q`, or `:β`, return the value for this transform. It may be in `st.settings`, or, if it is a default value, it is looked up.
"""
function getParameters(st, s)
    get(st.settings, s, default(s))
end

function default(s)
    if s == :dType
        Float32
    elseif s == :σ
        identity
    elseif s == :trainable
        false
    elseif s == :plan
        true
    elseif s == :init
        Flux.glorot_normal
    elseif s == :bias
        false
    elseif s == :convBoundary
        Sym()
    elseif s == :cw
        Morlet()
    elseif s == :averagingLayer
        false
    elseif s == :Q
        8
    elseif s == :boundary
        SymBoundary()
    elseif s == :averagingType
        Father()
    elseif s == :averagingLength
        4
    elseif s == :frameBound
        1
    elseif s == :p
        Inf
    elseif s == :β
        4
    end
end

import Base.size
function size(st::stFlux)
    l = st.mainChain[1]
    if typeof(l.fftPlan) <: Tuple
        sz = l.fftPlan[2].sz
    else
        sz = l.fftPlan.sz
    end
    es = originalSize(sz[1:ndims(l.weight[1])], l.bc)
    return es
end
