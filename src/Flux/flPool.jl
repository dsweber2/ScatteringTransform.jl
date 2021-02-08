maxpoolNd(x, poolSz) = maxpool(x, poolSz)
function maxpoolNd(x, poolSz::Tuple{<:Integer})
    shaped = reshape(x, (size(x, 1), 1, size(x)[2:end]...))
    subsampled = maxpool(shaped, (poolSz..., 1))
    ax = axes(subsampled)
    return subsampled[ax[1], 1, ax[3:end]...]
end




# extend maxpool to use rational types, in case pooling by 2x is too steep
import NNlib.maxpool

"""
    r = RationPool(resSize, k=2; nExtraDims=1, poolType = MeanPool)

slight extension of the Flux pooling methods to subsample at a rational rate
`resSize`. `k` is the window size over which to apply the pooling, and poolType
is the type of pooling, either MaxPool or MeanPool. nExtraDims counts the
number of dimensions uninvolved in the convolution; normally this is 2, the
last dimension for nExamples, and the penultimate for channels. You can expect
pooling to work on sizes up to 5 total dimensions.
"""
struct RationPool{A,B}
    m::A # should inheret from MaxPool
    resSize::B # a tuple of ints and rationals
end

function Base.show(io::IO, m::RationPool)
    print(io, "RationPool(windowSize=$(m.m.k), poolingRate=$(m.resSize))")
end

function RationPool(resSize::Tuple{Vararg{<:Union{<:Integer,Rational{<:Integer}},N}},
                    k=2; nExtraDims=2, poolType=MeanPool) where N
    effResSize = (resSize..., ntuple(ii -> 1 // 1, nExtraDims - 2)...)
    subBy = map(ki -> ((ki == 1) ? 1 : k), effResSize) # any non-trivial dim
    # should subsample at a rate of k
    m = poolType(subBy, pad=0, stride=1)
    # SamePad() means that sz = inputsize / stride
    RationPool{typeof(m),typeof(resSize)}(m, resSize)
end

RationPool(resSize::Union{<:Integer,Rational{<:Integer}}, k=3;
           nExtraDims=2) =
               RationPool((resSize,), k; nExtraDims=nExtraDims)

import Base:getindex
Base.getindex(X::RationPool, i::Union{AbstractArray,<:Integer}) = X.resSize[i]
import Base:ndims
ndims(r::MaxPool{N,M}) where {N,M} = N
ndims(r::MeanPool{N,M}) where {N,M} = N
nPoolDims(r::RationPool{A,<:Tuple{Vararg{<:Any,N}}}) where {A,N} = N
import Flux:outdims
outdims(r::RationPool) = 3
using Zygote:hook
function (r::RationPool)(x::AbstractArray{<:Any,N}) where N
    Nd = nPoolDims(r)
    Nneed = ndims(r.m) + 2
    extraDims = ntuple(ii -> 1, Nneed - N)
    partial = reshape(x, (size(x)[1:Nd]..., extraDims..., size(x)[Nd + 1:end]...))
    partial = r.m(partial)
    address = map(stopAtExactly_WithRate_FromSize_, size(partial)[1:nPoolDims(r)], r.resSize, size(x)[1:nPoolDims(r)])
    ax = axes(partial)
    endAxes = ax[Nd + Nneed - N + 1:end] # grab the stuff after extraDims
    return partial[address..., extraDims..., endAxes...]
end



"""
    poolSize(k, sizes)
if we're pooling at rates `k` in each dimension (e.g. `(3//2, 3//2)`), how many entries should we expect in the next layer
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
