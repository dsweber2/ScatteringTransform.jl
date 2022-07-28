"""
    stFlux(inputSize::NTuple{N}, m; trainable=false,
                             normalize=true, outputPool=2,
                             poolBy=3 // 2, σ=abs, flatten=false, kwargs...)
    stFlux(inputSize::NTuple{N}, m; trainable = false,
                        normalize = true, outputPool = 2,
                        poolBy= 3//2, σ=abs, scales=(4,4,4),
                        shearLevels=scales) where {N}

Create a scattering transform of depth `m` (which returns a `m+1` depth list of arrays) that subsamples at a rate of `poolBy` each layer, using `scales[i]` and `shearLevels[i]` at each layer. `normalize` means give each layer the same average weight per path, e.g. since the zeroth layer has one path, give it norm 1, if the first layer has 16 paths, give the sum across all first layer paths norm 16, etc. This is primarily for cases where the classification algorithm needs roughly the same order of
magnitude variance. `flatten` means return a result that is a matrix of size `(:,nExamples)` where `nExamples` is the last dimension of `inputSize`. `σ` is the nonlinearity used. Any additional keyword args `kwargs...` come from either the FourierFilterFlux [`waveletLayerConstructor`](https://dsweber2.github.io/FourierFilterFlux.jl/dev/constructors/#FourierFilterFlux.waveletLayer), or subsequently from ContinuousWavlets [`wavelet`](https://dsweber2.github.io/ContinuousWavelets.jl/dev/CWTConstruction/) constructor ContinuousWavelets.
"""
function stFlux(inputSize::NTuple{N}, m; trainable=false,
    normalize=true, outputPool=2,
    poolBy=3 // 2, σ=abs, flatten=false, kwargs...) where {N}
    # N determines the number of spatial dimensions
    Nd = N - 2
    if length(outputPool) == 1 # replicate for each dimension and layer
        outputPool = ntuple(k -> ntuple(i -> outputPool[1], Nd), m + 1)
    elseif length(outputPool) == m + 1 # replicate for each dimension
        outputPool = ntuple(k -> ntuple(i -> outputPool[k]), m + 1)
    elseif length(outputPool) == Nd
        outputPool = ntuple(k -> outputPool, m + 1)
    end
    if typeof(poolBy) <: Union{<:Rational,<:Integer}
        poolBy = map(ii -> RationPool((ntuple(i -> poolBy, N - 2)...,),
                nExtraDims=nPathDims(ii + 1) + 1), 1:m+1)
    end
    poolBy = makeTuple(m, poolBy)
    argsToEach = processArgs(m + 1, kwargs)

    listOfSizes = [(inputSize..., ntuple(i -> 1, max(i - 1, 0))...) for i = 0:m]
    interstitial = Array{Any,1}(undef, 3 * (m + 1) - 2)
    for i = 1:m
        # first transform
        interstitial[3*i-2] = dispatchLayer(listOfSizes[i], Val(Nd); σ=identity,
            argsToEach[i]...)
        nFilters = length(interstitial[3*i-2].weight)

        pooledSize = poolSize(poolBy[i], listOfSizes[i][1:Nd])
        # then throw away the averaging (we'll pick it up in the actual transform)
        # also, in the first layer, merge the channels into the first shearing,
        # since max pool isn't defined for arbitrary arrays
        if i == 1
            listOfSizes[i+1] = (pooledSize...,
                (nFilters - 1) * listOfSizes[1][Nd+1],
                listOfSizes[i][Nd+2:end]...)
            interstitial[3*i-1] = x -> begin
                ax = axes(x)
                return σ.(reshape(x[ax[1:Nd]..., ax[Nd+1][1:end-1], ax[(Nd+2):end]...],
                    (ax[1:Nd]..., listOfSizes[i+1][Nd+1], ax[end])))
            end
        else
            listOfSizes[i+1] = (pooledSize..., (nFilters - 1),
                listOfSizes[i][Nd+1:end]...)
            interstitial[3*i-1] = x -> begin
                ax = axes(x)
                return σ.(x[ax[1:Nd]..., ax[Nd+1][1:end-1], ax[Nd+2:end]...])
            end
        end
        # then pool
        interstitial[3*i] = poolBy[i]
    end

    # final averaging layer
    interstitial[3*m+1] = dispatchLayer(listOfSizes[end], Val(Nd);
        averagingLayer=true, σ=identity,
        argsToEach[end]...)
    chacha = Chain(interstitial...)
    outputSizes = ([(map(poolSize, outputPool[ii], x[1:Nd])..., x[(Nd+1):end]...) for (ii, x)
                    in
                    enumerate(listOfSizes)]...,)

    # record the settings used pretty kludgy
    settings = Dict(:outputPool => outputPool, :poolBy => poolBy, :σ => σ, :flatten => flatten, (argsToEach...)...)
    return stFlux{Nd,m,typeof(chacha),typeof(outputSizes),typeof(outputPool),typeof(settings)}(chacha, normalize,
        outputSizes, outputPool, settings)
end

function dispatchLayer(listOfSizes, Nd::Val{1}; varargs...)
    waveletLayer(listOfSizes; varargs...)
end

Base.size(a::Tuple{AbstractFFTs.Plan,AbstractFFTs.Plan}) = size(a[1])


# actually apply the transform
function (St::stFlux{Dimension,Depth})(x::T) where {Dimension,Depth,T<:AbstractArray}
    mc = St.mainChain.layers
    if ndims(x) < length(size(mc[1]))
        # the input is missing dimensions, so reshape
        targetSize = size(mc[1])
        x = reshape(x, (size(x)..., targetSize[length(targetSize)-ndims(x):end]...))
    end
    if size(x)[end] != getBatchSize(mc[1])
        res = breakAndAdapt(St, x)
    else
        res = applyScattering(mc, x, ndims(St), St, 0)
    end
    if get(St.settings, :flatten, false) # it may not be defined, in which case we don't do it
        batchSize = size(res[1])[end]
        return cat((reshape(x, (:, batchSize)) for x in res)..., dims=1)
    else
        return ScatteredOut(res, ndims(mc[1]))
    end
end

# adapt changes both the eltype and the container type, while I just want a different container type
function maybeAdapt(contType, x)
    if contType <: CuArray && !(typeof(x) <: CuArray)
        # should be a CuArray but isn't
        return cu(x)
    elseif contType <: Array && !(typeof(x) <: Array)
        # should be an Array but isn't
        return adapt(Array, x)
    else
        return x
    end
end

"""
extract x at adr in the last dimension, and make sure that it has a size of chunkSize
"""
function extractAddPadding(x, adr, chunkSize, N)
    justUsing = x[fill(Colon(), N)..., :, adr]
    if length(adr) < chunkSize
        actualSize = chunkSize - length(adr)
        return cat(justUsing, zeros(eltype(justUsing), size(x)[1:end-1]...,
                actualSize), dims=ndims(justUsing)), length(adr)
    else
        return justUsing, chunkSize
    end
end

trim(x, actualSize) = x[axes(x)[1:end-1]..., 1:actualSize]
function breakAndAdapt(St::stFlux{N,D}, x) where {N,D}
    mc = St.mainChain.layers
    chunkSize = size(mc[1].fftPlan)[end]
    nSteps = ceil(Int, (size(x)[end]) / chunkSize) # the first entry is taken care of already
    containerType = typeof(St.mainChain[1].weight[1])
    xAxes = axes(x)
    firstAddr = 1+0:min(size(x)[end], chunkSize)
    firstEx, actualSize = extractAddPadding(x, firstAddr, chunkSize, N) # x[xAxes[1:end-1]..., 1:chunkSize]
    # do the first beforehand to get the sizes
    out = applyScattering(mc, maybeAdapt(containerType, firstEx), ndims(St), St, 0)
    # create storage
    outputs = map(out -> zeros(eltype(out), size(out)[1:end-1]..., size(x)[end]...), out)
    # util to write out to outputs at location batchInds
    function writeOut!(out, batchInds, actualSize)
        for jj in 1:length(out)
            mA = maybeAdapt(typeof(x), out[jj])
            oAx = axes(outputs[jj])
            @views outputs[jj][oAx[1:end-1]..., batchInds] = trim(mA, actualSize)
        end
    end
    writeOut!(out, firstAddr, actualSize) # write the first batch out
    for ii = 2:nSteps
        out = 3
        # clear output to force garbage collection, otherwise the gpu may be full
        addr = 1+(ii-1)*chunkSize:min(size(x)[end], ii * chunkSize)
        tmpX, actualSize = extractAddPadding(x, addr, chunkSize, N)
        out = applyScattering(mc, maybeAdapt(containerType, tmpX), ndims(St), St, 0)
        writeOut!(out, addr, actualSize)
    end
    return outputs
end


function applyScattering(c::Tuple, x, Nd, st, M)
    res = first(c)(x)
    if typeof(first(c)) <: ConvFFT
        tmpRes = res[map(x -> Colon(), 1:Nd)..., end, map(x -> Colon(), 1:ndims(res)-Nd-1)...]
        # return a subsampled version of the output at this layer
        poolSizes = (st.outputPool[M+1]..., ntuple(i -> 1, ndims(tmpRes) - Nd - 2)...)
        r = RationPool(st.outputPool[M+1], nExtraDims=nPathDims(M + 1) + 1)
        if st.normalize
            tmpRes = normalize(r(real.(tmpRes)), Nd)
            apld = applyScattering(tail(c), res, Nd, st, M + 1)
            return (tmpRes, apld...)
        else
            tmpRes = r(real.(tmpRes))
            return (tmpRes,
                applyScattering(tail(c), res, Nd, st, M + 1)...)
        end
    else
        # this is either a reshaping layer or a subsampling layer, so no output
        return applyScattering(tail(c), res, Nd, st, M)
    end
end

applyScattering(::Tuple{}, x, Nd, st, M) = tuple() # all of the returns should
# happen along the way, not at the end

"""
    normedX = normalize(x, Nd)
normalize x over the dimensions Nd through ndims(x)-1, e.g. for a Nd=2, and x
that is 4D, then norm(x[:,:,:,j], 2) ≈ size(x,3)
"""
function normalize(x, Nd)
    n = ndims(x)
    totalThisLayer = prod(size(x)[(Nd+1):(n-1)])
    ax = axes(x)
    buf = Zygote.Buffer(x)
    for i = 1:size(x)[end]
        xSlice = x[ax[1:(n-1)]..., i]
        thisExample = totalThisLayer / (sum(abs.(xSlice) .^ 2)) .^ (0.5f0)
        if isnan(thisExample) || thisExample ≈ Inf
            buf[ax[1:(n-1)]..., i] = xSlice
        else
            buf[ax[1:(n-1)]..., i] = xSlice .* thisExample
        end
    end
    return copy(buf)
end

normalize(sct::ScatteredOut, Nd) = ScatteredOut(map(x -> normalize(x, Nd),
        sct.output), sct.k)

normalize(sct::ScatteredFull, Nd) = ScatteredFull(sct.m, sct.k, sct.data,
    map(x -> normalize(x, Nd),
        sct.output))
