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
    relevantLocs = [sum(netSizes[1:(i)]) for i in 0:length(netSizes)]
    batchSize = size(res[1])[end]
    singleExampleSize = sum(netSizes)
    output = adapt(typeof(res[1]), zeros(singleExampleSize, batchSize))
    output = cat([reshape(x, (netSizes[i], batchSize)) for (i,x) in enumerate(res)]..., dims=1)
    # for (i,x) in enumerate(scatRes)
    #     indices = (1+relevantLocs[i]):relevantLocs[i+1]
    #     output[indices, :] = reshape(x, (netSizes[i], batchSize))
    # end
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
    nExamples = size(toRoll)[2:end]
    rolled = ([adapt(typeof(toRoll), zeros(eltype(toRoll),
                                          sz[1:Nd+nPathDims(ii)]...,
                                          nExamples...)) for (ii,sz) in
              enumerate(oS)]...,);

    locSoFar = 0
    for (ii, x) in enumerate(rolled)
        szThisLayer = oS[ii][1:Nd+nPathDims(ii)]
        println(szThisLayer)
        totalThisLayer = prod(szThisLayer)
        range = (locSoFar + 1):(locSoFar + totalThisLayer)
        addresses = (szThisLayer..., nExamples...)
        rolled[ii][:] = reshape(toRoll[range, :], addresses)
        locSoFar += totalThisLayer
    end
    return ScatteredOut(rolled, Nd)
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
