abstract type stType end
struct fullType <: stType end
struct decreasingType <: stType end

(::Colon)(a::Tuple, b::Tuple) = (:).(a,b)
(::Colon)(a::Tuple, b::Tuple, c::Tuple) = (:).(a,b,c)


@doc """
      n, q, dataSizes, outputSizes, resultingSize = calculateSizes(layers::stParallel{K,1}, outputSubsample, Xsize; totalScales = [-1 for i=1:depth(layers)+1]) where {K}

  * n is a list of the sizes of a single example. It is (m+1)×dataDim
one dimension at a time
  * q is a list of the number of scales, as used in 1D
  * datasizes is a list of tuples of sizes encountered in each layer
  * outputsizes is the same for the output arrays
  * resultingsize is a list of the output size as a result of subsampling
"""
function calculateSizes(layers::stParallel{K,1},
                        outputSubsample, Xsize; totalScales = [-1
                                                               for
                                                               i=1:depth(layers)+1],
                        percentage = .9) where {K}
    n = Int.(sizes(bsplineType(), layers.subsampling, Xsize[1:1]))
    q = getQ(layers, n, totalScales)
    dataSizes = [Int.([n[i], q[i], Xsize[2:end]...]) for i=1:depth(layers)+1]
    outputSizes = [Int.([n[i+1], q[i], Xsize[2:end]...]) for i=1:depth(layers)+1]
    if outputSubsample[1] > 1
        resultingSize = zeros(Int, depth(layers)+1)
        resultingSubsampling = zeros(depth(layers)+1)
        # subsampling limited by the second entry
        for (i,x) in enumerate(outputSizes)
            proposedLevel = floor(Int, x[1]/outputSubsample[1])
            if proposedLevel < outputSubsample[2]
                # the result is smaller than the floor
                proposedLevel = outputSubsample[2]
            end
            resultingSize[i] = Int(proposedLevel)
        end
    elseif outputSubsample[2] > 1
        resultingSize = outputSubsample[2]*ones(Int64,depth(layers)+1)
    else
        resultingSize = [x[1] for x in outputSizes]
    end
    return (n, q, dataSizes, outputSizes, resultingSize)
end

"""

get the sizes of a single example in each layer
"""
function getNs(Xsize, layers::stParallel{K,1}) where K
    return Int.(sizes(bsplineType(), layers.subsampling, Xsize[(end):end]))
end

@doc """
  q = getQ(layers, n, totalScales; product=true)
calculate the total number of entries in each layer
"""
function getQ(layers::stParallel{K,1}, n, totalScales; product=true) where {K}
    # first just the number of new scales
    q = [numScales(layers.shears[i], n[i])-1 for i=1:depth(layers)+1]
    # then a product over all previous
    q = [(isnan(totalScales[i]) || totalScales[i]<=0) ? q[i] : totalScales[i] for i=1:depth(layers)+1]
    if product
        q = [prod(q[1:i-1]) for i=1:depth(layers)+1]
        return q
    else
        return [1; reverse(q)[2:end]...]
    end
end

getQ(layers, Xsize; product=true) = getQ(layers, Int.(sizes(bspline, layers.subsampling, Xsize[(end):end]))
, [-1 for i=1:depth(layers)+1]; product=product)

"""

a list of the paths in a given transform. E.g. if the first layer has 13 scales and the second layer has 9, then this gives [[], [13], [9,13]].
"""
function getListOfScaleDims(layers,n,totalScales=[-1 for i=1:depth(layers) +1])
    nScalesEachLayer = getQ(layers, n, totalScales; product=false)
    addedLayers = [nScalesEachLayer[end+2-i:end] for i=1:depth(layers) + 1]
end
"""
return a list of locations for iterating over layer i in the same order as the
transform (depth first, so early layers are the slow changing variable)
"""
function indexOverScales(addedLayers,i)
    if i!= 1
        axe = [1:nScales for nScales in addedLayers[i]]
        Iterators.product(axe...)
    else
        return 1
    end
end

function linearFromTuple(js,nScales)
    if length(js)>1
        js[1]-1 + nScales[1] * linearFromTuple(js[2:end],nScales[2:end])
    else
        return js[1]-1
    end
end
function getLastScales(dataDim,daughters, shears)
        size(daughters,2)
end









"""
    createFFTPlans(layers::stParallel{K, N}, dataSizes; verbose=false, T=Float32, iscomplex::Bool=false)

to be called after distributed. returns a 2D array of futures; the i,jth entry has a future (referenced using fetch) for the fft to be used by worker i in layer j.
"""
function createFFTPlans(layers::stParallel{<:Any, N}, dataSizes;
                        T=Float32, iscomplex::Bool=false) where {N}
    @debug "starting to create plans at all"
    @debug "" nwork=nworkers()
    nPlans = getNumPlans(layers)
    FFTs = Array{Future,2}(undef, nworkers(), depth(layers)+1)
    for i=1:nworkers(), j=1:depth(layers)+1
        @debug "i=$(i), j=$(j)"
        if length(layers.n) >= 2
            FFTs[i, j] = remotecall(createRemoteFFTPlan, i,
                                    dataSizes[j][[1 3:end]], T,
                                    iscomplex, nPlans)
        else
            fftPlanSize = (dataSizes[j][1]*2, dataSizes[j][3:end]...)
            # in the last layer
            @debug "plans are going to be size $(fftPlanSize)"
            FFTs[i, j] = remotecall(createRemoteFFTPlan, i, fftPlanSize,
                                    T, iscomplex, nPlans)
        end
    end

    return FFTs
end

# only in the 1D case where we're using either Morlet or Paul wavelets do we
# need both a rfft and an fft
function getNumPlans(layers::stParallel{<:Any,1})
    if typeof(layers.shears[1].waveType) <: Union{Morlet, Paul}
        return 2
    else
        return 1
    end
end

"""
the 1D version
"""
function createRemoteFFTPlan(sizeOfArray, T, iscomplex, nPlans::Int)
    if iscomplex
        return plan_fft!(zeros(Complex{T}, sizeOfArray...), (1,), flags =
                         FFTW.PATIENT)
    elseif nPlans==1
        return plan_rfft(zeros(T, sizeOfArray...), (1,), flags =
                         FFTW.PATIENT)
    elseif nPlans==2
        return (plan_rfft(zeros(T, sizeOfArray...), (1,), flags =
                          FFTW.PATIENT),
                plan_fft!(zeros(Complex{T}, sizeOfArray...), (1,), flags =
                         FFTW.PATIENT))
    end
end

function remoteMult(x::Future,y)
    fetch(x)*y
end
remoteMult(y,x::Future) = remoteMultiply(x,y)

function remoteDiv(x::Future,y)
    fetch(x) / y
end
remoteDiv(y,x::Future) = remoteMultiply(x,y)









function computeAllWavelets(layers, N; outputSubsample=(-1,-1),totalScales =
                            [-1 for i=1:depth(layers)+1], percentage=.9)
    n, q, dataSizes, outputSizes, resultingSize = 
        ScatteringTransform.calculateSizes(layers, outputSubsample, N, 
                                           totalScales = totalScales, 
                                           percentage = percentage)
    daughters, ω = computeWavelets(n[1],
                                   layers.shears[1];
                                   nScales=-1)
    allDaughters = Array{Array{eltype(daughters),2},1}(undef, length(n))
    allDaughters[1] = daughters
    for (ii,x) in enumerate(n[1:end-1])
        daughters, ω = computeWavelets(n[ii],
                                       layers.shears[ii];
                                       nScales=-1)
        allDaughters[ii] = daughters
    end
    daughters, ω = computeWavelets(n[end],
                                   layers.shears[end];
                                   nScales=0)
    allDaughters[end] = daughters
    return allDaughters
end
function plotAllWavelets(allDaughters)
    return plot(plot([allDaughters[1] sum(allDaughters[1],
                                          dims=2)], legend=false, 
                     title= "layer 1"),
                heatmap([allDaughters[1] sum(allDaughters[1], dims=2)]),
                plot([allDaughters[2] sum(allDaughters[2],
                                          dims=2)],legend=false, 
                     title= "layer 2"), 
                heatmap([allDaughters[2] sum(allDaughters[2], dims=2)]),
                plot(allDaughters[3][:,1],legend=false, title= "output last"),
                heatmap(allDaughters[3][:,1:1]),
                layout = (3,2)
                )
end
