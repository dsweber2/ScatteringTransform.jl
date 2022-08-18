maxpoolNd(x, poolSz) = maxpool(x, poolSz)
function maxpoolNd(x, poolSz::Tuple{<:Integer})
    shaped = reshape(x, (size(x, 1), 1, size(x)[2:end]...))
    subsampled = maxpool(shaped, (poolSz..., 1))
    ax = axes(subsampled)
    return subsampled[ax[1], 1, ax[3:end]...]
end




# extend maxpool to use rational types, in case pooling by 2x is too steep
import NNlib.maxpool

@doc """
    RationPool{A,B}
An extension of Flux's `MaxPool` and `MeanPool` to subsampling by rational amounts as well. Don't construct directly. Has fields `r.m`, which is the pooling operator as implemented in Flux, and `r.resSize`, which gives the subsampling rates, either as `Integer`, `Rational`, or tuples of these.

It works by first applying the given pooling operator, but with a step size of 1, and then keeping `p` out of `q` entries, where `p` is the numerator of the rational rate, and `q` is the denominator.
"""
struct RationPool{A,B}
    m::A # should inheret from MaxPool
    resSize::B # a tuple of ints and rationals
end

function Base.show(io::IO, m::RationPool)
    print(io, "RationPool(windowSize=$(m.m.k), poolingRate=$(m.resSize))")
end

@doc """
    RationPool(resSize, k=2; nExtraDims=1, poolType = MeanPool)

Construct a `RationPool` instance, which is a slight extension of the Flux pooling methods to subsample at a rational rate `resSize`, which can vary by channel. For example, for a 2D input, `resSize` could be `(2,3//2)`, `3//2` (equivalent to `(3//2, 3//2)`), or `(5//3, 5//3)`.

`k` is the window size of the pooling, and `poolType` is the type of pooling, either `MaxPool` or `MeanPool`. `nExtraDims` counts the number of dimensions uninvolved in the convolution; normally this is 2, as in `(..., nChannels, nExamples)`. You can expect pooling to work for sizes up to 5 total dimensions.
"""
function RationPool(resSize::NTuple{N,Union{<:Integer,Rational{<:Integer}}}, k=2; nExtraDims=2, poolType=MeanPool) where {N}
    effResSize = (resSize..., ntuple(ii -> 1 // 1, min(nExtraDims - 2, 5))...)
    subBy = map(ki -> ((ki == 1) ? 1 : k), effResSize) # any non-trivial dim
    # should subsample at a rate of k
    m = poolType(subBy, pad=0, stride=1)
    # SamePad() means that sz = inputsize / stride
    RationPool{typeof(m),typeof(resSize)}(m, resSize)
end

RationPool(resSize::Union{<:Integer,Rational{<:Integer}}, k=3;
    nExtraDims=2) =
    RationPool((resSize,), k; nExtraDims=nExtraDims)

import Base: getindex
Base.getindex(X::RationPool, i::Union{AbstractArray,<:Integer}) = X.resSize[i]
import Base: ndims
"""
    ndims(r::MaxPool{N,M})
    ndims(r::MeanPool{N,M})

return the dimension `N` of the input signal
"""
ndims(r::MaxPool{N,M}) where {N,M} = N
ndims(r::MeanPool{N,M}) where {N,M} = N
nPoolDims(r::RationPool{A,<:NTuple{N,<:Any}}) where {A,N} = N
import Flux: outdims
outdims(r::RationPool) = 3
using Zygote: hook
function (r::RationPool)(x::AbstractArray{<:Any,N}) where {N}
    Nd = nPoolDims(r)
    Nneed = ndims(r.m) + 2
    if N <= Nneed
        # x needs to be slightly padded
        extraDims = ntuple(ii -> 1, Nneed - N)
        partial = reshape(x, (size(x)[1:Nd]..., extraDims..., size(x)[Nd+1:end]...))
    else
        # x is too big and needs to be temporarily reshaped
        extraDims = ntuple(ii -> 1, 2)
        partial = reshape(x, (size(x)[1:Nd]..., extraDims..., prod(size(x)[Nd+1:end])))
    end
    partial = r.m(partial)
    address = map(stopAtExactly_WithRate_FromSize_, size(partial)[1:nPoolDims(r)], r.resSize, size(x)[1:nPoolDims(r)])
    ax = axes(partial)
    if N <= Nneed
        endAxes = ax[Nd+Nneed-N+1:end] # grab the stuff after extraDims
        return partial[address..., extraDims..., endAxes...]
    else
        return reshape(partial[address..., extraDims..., ax[Nd+3:end]...], (length.(address)..., size(x)[Nd+1:end]...))
    end
end



"""
    poolSize(k::RationPool, sizes)
    poolSize(k, sizs)
Return the number should we expect in the output with pooling rates `k` in each dimension (e.g. `(3//2, 3//2)` or `RationPool((3//2,3//2))`).
"""
function poolSize(r::RationPool, sizes)
    resSize = [x for x in r.resSize]
    [poolSingle(resSize[i], sizes[i]) for i in 1:length(resSize)]
end

function poolSize(kks, sizs)
    map(poolSingle, kks, sizs)
end

function poolSingle(kk, siz)
    return length(stopAtExactly_WithRate_(siz, kk))
end

function stopAtExactly_WithRate_(i, subBy)
    tmp = round.(Int, range(1, stop=i, length=round(Int, i / subBy)))
    return tmp
end
function stopAtExactly_WithRate_FromSize_(i, subBy, orig)
    tmp = round.(Int, range(1, stop=i, length=round(Int, orig / subBy)))
    return tmp
end
