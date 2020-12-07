# how should we apply a function recursively to the stFlux type? to each part of the chain, of course
import FourierFilterFlux.functor
import Flux.functor
function functor(stf::stFlux{Dimension, Depth, ChainType, D, E,F}) where {Dimension, Depth, ChainType, D, E,F}
    return (stf.mainChain...,), y -> begin
        stFlux{Dimension, Depth, typeof(Chain(y...)), D,E,F}(Chain(y...),stf.normalize, stf.outputSizes, stf.outputPool, stf.settings)
    end
end


"""
wave1, wave2, wave3, ... = getWavelets(sc::stFlux)

just a simple util to get the wavelets from each layer
"""
function getWavelets(sc::stFlux)
    map(x->x.weight, filter(x->(typeof(x) <: ConvFFT), sc.mainChain.layers)) # filter to only
    # have ConvFFTs, and then return the wavelets of those
end

"""
    output = flatten(scatRes)
given a scattered, it produces a single vector containing the entire transform in order, i.e. the same format as output by thinSt.
"""
function flatten(scatRes::S) where S <: Scattered
    res = scatRes.output
    netSizes = [prod(size(r)[1:end-1]) for r in res]
    batchSize = size(res[1])[end]
    singleExampleSize = sum(netSizes)
    output = cat((reshape(x, (netSizes[i], batchSize)) for (i,x) in enumerate(res))..., dims=1)
    return output
end
flatten(scatRes) = scatRes


"""
    rolled = roll(toRoll, st::stFlux)
Given the output of a scattering transform and something with the same number
of entries but in an array that is NCoeffs×extraDims, roll up the output
into an array of arrays like the scattered.
    rolled = roll(toRoll, st::stParallel; percentage=.9, outputSubsample=(-1,-1))
there is also a version for the parallel transform; since `percentage` and 
`outputSubsample` are separate variables, they must also be input.
"""
function roll(toRoll, st::stFlux)
    Nd = ndims(st)
    oS = st.outputSizes
    roll(toRoll, oS, Nd)
end

function roll(toRoll, stOutput::S) where S<:Scattered
    Nd = ndims(stOutput)
    oS = map(size,stOutput.output)
    return roll(toRoll, oS, Nd)
end

function roll(toRoll, oS::Tuple, Nd)
    nExamples = size(toRoll)[2:end]
    rolled = ([adapt(typeof(toRoll), zeros(eltype(toRoll),
                                          sz[1:Nd+nPathDims(ii)]...,
                                          nExamples...)) for (ii,sz) in
              enumerate(oS)]...,);

    locSoFar = 0
    for (ii, x) in enumerate(rolled)
        szThisLayer = oS[ii][1:Nd+nPathDims(ii)]
        totalThisLayer = prod(szThisLayer)
        range = (locSoFar + 1):(locSoFar + totalThisLayer)
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
            return pathLocs(ii-1,Tuple(CartesianIndices(szThisLayer)[1+loc-locSoFar]))
        end
        locSoFar += totalThisLayer
    end
end

"""
given a scattered output, make a list that gives the largest value on each path
"""
function importantCoords(scatRes)
    return [dropdims(maximum(abs.(x),dims=(1,2)), dims=(1,2)) for x in scatRes]
end



"""
if the batch size is off, we don't want to suddenly drop performance. Split it up.
"""
function batchOff(stack, x, batchSize)
    nRounds = ceil(Int, size(x,4)//batchSize)
    firstRes = stack(x[:,:,:,1:batchSize]);
    result = cu(zeros(size(firstRes)[1:end-1]..., size(x)[end]))
    result[:,1:batchSize] = firstRes
    for i=2:(nRounds-1)
        result[:, 1 + (i-1)*batchSize:(i*batchSize)] = stack(x[:,:,:,1 + (i-1)*batchSize:(i*batchSize)])
    end
    result[:, (1+(nRounds-1)*batchSize):end] = stack(cat(x[:, :, :,
                                                           (1+(nRounds-1)*batchSize):end],
                                                         cu(zeros(size(x)[1:3]...,
                                                                  nRounds*batchSize
                                                                  - size(x, 4))),
                                                         dims=4))[:, 1:(size(x,4)-(nRounds-1)*batchSize)]
    return result
end
