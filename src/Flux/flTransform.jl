"""
    stFlux(inputSize::NTuple{N}, m; trainable = false, normalize = true, 
                        outputPool = 2, poolBy= 3//2, σ=abs, 
                        scales=(8,8,8), shearLevels=scales/2, cws=WT.morl,
                        averagingLengths=[4,4,4], 
                        decreasings=[2,2,2]) where {N}
    stFlux(inputSize::NTuple{N}, m; trainable = false,
                        normalize = true, outputPool = 2,
                        poolBy= 3//2, σ=abs, scales=(4,4,4),
                        shearLevels=scales) where {N} 

Create a scattering transform of depth m (which returns a m+1 depth list of
arrays) that subsamples at a rate of poolBy each layer, using scales[i] and
`shearLevels[i]` at each layer. Normalize means give each layer the same average
weight per path, e.g. since the zeroth layer has one path, give it norm 1, if
the first layer has 16 paths, give it norm 16, etc. This is primarily for
cases where the classification algorithm needs roughly the same order of
magnitude variance.
"""
function stFlux(inputSize::NTuple{N}, m; trainable = false,
                             normalize = true, outputPool = 2,
                             poolBy= 3//2, σ=abs, kwargs...) where {N} 
    # N determines the number of spatial dimensions
    Nd = N - 2;
    if length(outputPool) == 1 # replicate for each dimension and layer
        outputPool = ntuple(k->ntuple(i->outputPool[1], Nd), m+1)
    elseif length(outputPool) == m+1 # replicate for each dimension
        outputPool = ntuple(k-> ntuple(i->outputPool[k]), m+1)
    elseif length(outputPool) == Nd
        outputPool = ntuple(k->outputPool, m+1)
    end
    if typeof(poolBy) <: Union{<:Rational, <:Integer}
        poolBy = map(ii->RationPool((ntuple(i->poolBy, N-2)...,), 
                            nExtraDims = nPathDims(ii+1)+1),1:m+1)
    end
    poolBy = makeTuple(m, poolBy)
    argsToEach = processArgs(m+1, kwargs)

    listOfSizes = [(inputSize..., ntuple(i->1, max(i-1, 0))...) for i=0:m]
    interstitial = Array{Any,1}(undef, 3*(m+1)-2)
    for i=1:m
        # first transform
        interstitial[3*i - 2] = dispatchLayer(listOfSizes[i], Val(Nd); σ=identity,
                                              argsToEach[i]...)
        nFilters = size(interstitial[3*i - 2].weight)[end]
        
        pooledSize = poolSize(poolBy[i], listOfSizes[i][1:Nd])
        # then throw away the averaging (we'll pick it up in the actual transform)
        # also, in the first layer, merge the channels into the first shearing,
        # since max pool isn't defined for arbitrary arrays
        if i==1
            listOfSizes[i+1] = (pooledSize...,
                                (nFilters-1)*listOfSizes[1][Nd+1],
                                listOfSizes[i][Nd+2:end]...)
            interstitial[3*i-1] = x -> begin
                ax = axes(x)
                return σ.(reshape(x[ax[1:Nd]..., ax[Nd+1][1:end-1], ax[(Nd+2):end]...],
                               (ax[1:Nd]..., listOfSizes[i+1][Nd+1], ax[end])))
            end
        else
            listOfSizes[i+1] = (pooledSize..., (nFilters-1),
                                listOfSizes[i][Nd+1:end]...)
            interstitial[3*i-1] = x-> begin
                ax = axes(x)
                return σ.(x[ax[1:Nd]..., ax[Nd+1][1:end-1], ax[Nd+2:end]...])
            end
        end
        # then pool
        interstitial[3*i] = poolBy[i]
    end

    # final averaging layer
    interstitial[3*m + 1] = dispatchLayer(listOfSizes[end], Val(Nd);
                                        averagingLayer=true, σ=identity, 
                                        argsToEach[end]...)
    chacha = Chain(interstitial...)
    outputSizes = ([(map(poolSize, outputPool[ii], x[1:Nd])..., x[(Nd+1):end]...) for (ii,x)
                    in enumerate(listOfSizes)]...,)

    # record the settings used pretty kludgy
    settings = (:outputPool=>outputPool, :poolBy=>poolBy, :σ=>σ, argsToEach...)
    return stFlux{Nd, m, typeof(chacha),
                  typeof(outputSizes), typeof(outputPool),
                  typeof(settings)}(chacha, normalize,
                                    outputSizes, outputPool, settings)
end

function dispatchLayer(listOfSizes, Nd::Val{1}; varargs...)
    waveletLayer(listOfSizes; varargs...)
end

function dispatchLayer(listOfSizes, Nd::Val{2}; varargs...)
    shearingLayer(listOfSizes; varargs...)
end




# actually apply the transform
function (st::stFlux)(x::T) where {T<:AbstractArray}
    mc = st.mainChain.layers
    res = applyScattering(mc, x, ndims(st), st, 0)
    return ScatteredOut(res, ndims(mc[1]))
end


function applyScattering(c::Tuple, x, Nd, st, M)
    res = first(c)(x)
    if typeof(first(c)) <: ConvFFT
        tmpRes = res[map(x->Colon(), 1:Nd)..., end, map(x->Colon(), 1:ndims(res)-Nd-1)...]
        # return a subsampled version of the output at this layer
        poolSizes = (st.outputPool[M+1]..., ntuple(i->1, ndims(tmpRes)-Nd-2)...)
        r = RationPool(st.outputPool[M+1], nExtraDims = ndims(tmpRes) - Nd)
        if st.normalize
            tmpRes = normalize(r(real.(tmpRes)), Nd)
            apld = applyScattering(tail(c), res, Nd, st, M+1)
            return (tmpRes, apld...)
        else
            tmpRes = r(real.(tmpRes))
            return (tmpRes,
                    applyScattering(tail(c), res, Nd, st, M+1)...)
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
    for i=1:size(x)[end]
        xSlice = x[ax[1:(n-1)]..., i]
        thisExample = totalThisLayer / (sum(abs.(xSlice).^2)).^(0.5f0)
        if isnan(thisExample) || thisExample≈Inf
            buf[ax[1:(n-1)]..., i] = xSlice
        else
            buf[ax[1:(n-1)]..., i] = xSlice .* thisExample
        end
    end
    return copy(buf)
end

normalize(sct::ScatteredOut, Nd) = ScatteredOut(sct.m, sct.k,
                                                map(x->normalize(x,Nd),
                                                    sct.output))

normalize(sct::ScatteredFull, Nd) = ScatteredFull(sct.m, sct.k, sct.data,
                                                  map(x->normalize(x,Nd),
                                                      sct.output))
