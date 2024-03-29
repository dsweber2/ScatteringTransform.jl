"""
    Scattered{T,N}
The abstract parent type for `ScatteredOut` and `ScatteredFull`. `T` gives the element type for the matrices, while `N` gives one plus the depth of the scattering transform (so for two layers it is three).
"""
abstract type Scattered{T,N} end
# TODO: tests for the data collators and the subsampling
# TODO: Write a version of this that accomodates things that are too big to hold in memory
# TODO: write a version of this that uses sparse matrices
# TODO: adapt for complex input and/or complex wavelets
struct ScatteredFull{T,N} <: Scattered{T,N}
    m::Int # number of layers, counting the zeroth layer
    k::Int # the meta-dimension of the signals (should be either 1 or 2)
    data # original last dimension is time, new path/frequency
    output # The final averaged results; this is the output from the entire system
    function ScatteredFull{T,N}(m, k, data, output) where {T<:Number,N}
        @assert N > k # The size of the arrays must be at least 1 larger than the
        # input dimension
        m = Int64(m)
        k = Int64(k)
        # check data
        singlentry = eltype(data[1])
        arrayType = eltype(data)
        @assert arrayType <: AbstractArray{singlentry,N}
        @assert size(data, 1) == m + 1

        # check output
        singlentry = eltype(output[1])
        arrayType = eltype(output)
        @assert arrayType <: AbstractArray{singlentry,N}
        @assert size(output, 1) == m + 1
        new(m, k, data, output)
    end
end

function ScatteredFull(data, output, k=1)
    @assert eltype(data) == eltype(output)
    ScatteredOut{eltype(output),length(output)}(length(output) - 1, k,
        data, output)
end

"""
A simple wrapper for the results of the scattering transform. Its one field
`result` contains a tuple of the results from each layer, in the order zero,
one, two. Say we have an example `s`. You can access layer i by `s[i]`, so
`s[0]` gives the zeroth layer
"""
struct ScatteredOut{T,N} <: Scattered{T,N}
    m::Int# number of layers, counting the zeroth layer
    k::Int# the meta-dimension of the signals (should be either 1 or 2)
    output
end

function ScatteredOut(output, k=1)
    ScatteredOut{eltype(output),length(output)}(length(output) - 1, k,
        output)
end

arrayType(::ScatteredOut{T,N}) where {T,N} = T
noutputs(::ScatteredOut{T,N}) where {T,N} = N


@doc """
      ScatteredOut(m, k, fixDim::Array{<:Real, 1}, n::Array{<:Real, 2}, q::Array{<:Real, 1}, T::DataType)

The resulting output of a scattering transform. `m` is the number of layers, while `k` is the actual signal dimension. `fixDim` is the size of any auxillary dimensions for the input (for example if the input is a list of `nEx` examples of length 100 each, then `fixDim=(nEx,)`, while `n` is a list of the subsampled signal sizes and `q` is a list of the paths, as generated by the scatteringTransform). Finally, the element type is `T`. Initializes all entries to zero.

      ScatteredOut(output, k = 1)
A less involved constructor given just a list or tuple of the output from each layer. `k` gives the signal dimension, as above, with the default that `k=1`.
  """
function ScatteredOut(m, k, fixDim, n, q, T)
    @assert m + 1 == size(n, 1)
    @assert m + 1 == length(q)
    @assert k == size(n, 2)
    fixDim = Int.(fixDim)
    n = Int.(n)
    q = Int.(q)
    N = k + length(fixDim) + 1
    output = [zeros(T, n[i, :]..., prod(q[1:(i-1)] .- 1), fixDim...) for i = 1:m+1]
    return ScatteredOut{T,N}(m, k, output)
end

function ScatteredFull(m, k, fixDim::Array{<:Real,1}, n::Array{<:Real,2},
    q::Array{<:Real,1}, T)
    @assert m + 1 == size(n, 1)
    @assert m + 1 == length(q)
    @assert k == size(n, 2)
    fixDim = Int.(fixDim)
    n = Int.(n)
    q = Int.(q)
    N = k + length(fixDim) + 1
    data = [zeros(T, n[i, :]..., prod(q[1:i] .- 1), fixDim...) for i = 1:m+1]
    output = [zeros(T, n[i, :]..., prod(q[1:(i-1)] .- 1), fixDim...) for i = 1:m+1]
    return ScatteredFull{T,N}(m, k, data, output)
end


import Base.==, Base.+, Base.-, Base.broadcast, Base.length, Base.size, Base.iterate, Base.broadcastable

==(a::ScatteredOut, b::ScatteredOut) = all(a.output .== b.output)
==(a::ScatteredFull, b::ScatteredFull) = all(a.data .== b.data) && all(a.output .== b.output)
function +(a::ScatteredOut, b::ScatteredOut)
    @assert a.m == b.m
    @assert a.k == b.k
    ScatteredOut{arrayType(a),ndims(a)}(a.m, a.k, map((x, y) -> x .+ y, a.output, b.output))
end
function +(a::ScatteredFull, b::ScatteredFull)
    @assert a.m == b.m
    @assert a.k == b.k
    ScatteredFull(a.m, a.k, map((x, y) -> x .+ y, a.data, b.data), map((x, y) -> x .+ y, a.output, b.output))
end
function -(a::ScatteredOut, b::ScatteredOut)
    @assert a.m == b.m
    @assert a.k == b.k
    ScatteredOut{arrayType(a),ndims(a)}(a.m, a.k, map((x, y) -> x - y, a.output, b.output))
end
function -(a::ScatteredFull, b::ScatteredFull)
    @assert a.m == b.m
    @assert a.k == b.k
    ScatteredFull(a.m, a.k, map((x, y) -> x - y, a.data, b.data), map((x, y) -> x - y, a.output, b.output))
end

import Base.iterate, Base.broadcastable, Base.similar
broadcast(f, a::ScatteredOut, varargs...) =
    ScatteredOut(a.m, a.k, map(x -> broadcast(f, x, varargs...), a.output))
broadcast(f, a::ScatteredFull, varargs...) =
    ScatteredFull(a.m, a.k, map(x -> broadcast(f, x, varargs...), a.data),
        map(x -> broadcast(f, x, varargs...), a.output))
broadcastable(a::Scattered) = a.output
iterate(a::Scattered) = iterate(a.output)
iterate(a::Scattered, b) = iterate(a.output, b)
length(sct::Scattered) = length(sct.output) - 1
"""
    ndims(sct::Scattered)
return the input dimension size (also given by `sct.k`)
"""
ndims(sct::Scattered) = sct.k
size(sct::ScatteredOut) = map(x -> size(x), sct.output)
size(sct::ScatteredFull) = map(x -> size(x), sct.data)
similar(sct::ScatteredOut) = ScatteredOut(map(x -> similar(arrayType(sct), axes(x)), sct.output), sct.k)
similar(sct::ScatteredFull) = ScatteredFull(map(x -> similar(x), sct.data), map(x -> similar(x), sct.output), sct.k)

import Statistics.mean
function mean(a::ScatteredOut)
    ScatteredOut(a.m, a.k, map(x -> mean(x, dims=ndims(x)), a.output))
end
import Base.cat

function cat(sc::ScatteredOut, sc1::ScatteredOut, sc2::ScatteredOut...; dims=-1)
    @assert !any([sc.m != sc1.m, (sc.m .!= map(s -> s.m, sc2))])# check they're
    @assert !any([sc.k != sc1.k, (sc.k .!= map(s -> s.k, sc2))])# all equal
    outputs = map(x -> x.output, (sc, sc1, sc2...))
    output = map(x -> cat(x..., dims=ndims(x[1])), zip(outputs...))
    return ScatteredOut(sc.m, sc.k, tuple(output...))
end
function cat(sc::ScatteredFull, sc1::ScatteredFull, sc2::ScatteredFull...; dims=-1)
    @assert !any([sc.m != sc1.m, (sc.m .!= map(s -> s.m, sc2))])# check they're
    @assert !any([sc.k != sc1.k, (sc.k .!= map(s -> s.k, sc2))])# all equal
    outputs = map(x -> x.output, (sc, sc1, sc2...))
    output = map(x -> cat(x..., dims=ndims(x[1])), zip(outputs...))
    datas = map(x -> x.data, (sc, sc1, sc2...))
    data = map(x -> cat(x..., dims=ndims(x[1])), zip(datas...))
    return ScatteredFull(sc.m, sc.k, tuple(data...), tuple(output...))
end


function Base.show(io::IO, s::ScatteredOut{T,N}) where {T,N}
    print(io, "ScatteredOut{$T,$N} $(s.k) dim. OutputSizes:")
    outputSizes = map(x -> size(x), s.output)
    for x in outputSizes
        print(io, "\n    $(x)")
    end
end

function Base.show(io::IO, s::ScatteredFull{T,N}) where {T,N}
    print(io, "ScatteredFull{T,N} $(s.k) dim. Data sizes:")
    dataSizes = map(x -> size(x), s.data)
    for x in dataSizes
        print(io, "\n    $(x)")
    end
    print(io, "\nOutputSizes:")
    outputSizes = map(x -> size(x), s.output)
    for x in outputSizes
        print(io, "\n    $(x)")
    end
end

# pathLocs constructors that use a Scattered type
function pathLocs(s::Scattered)
    return pathLocs{length(s.output)}(map(x -> axes(x), s.output))
end

function pathLocs(ii, s::Scattered)
    return pathLocs{length(s.output)}(map(x -> axes(x), s.output[ii]))
end

# access methods
import Base: getindex
Base.getindex(X::Scattered, i::Union{Tuple,<:AbstractArray}) = X.output[i.+1]
function Base.getindex(X::Scattered, ::Colon)
    if ndims(X.output[1]) > ndims(X)
        nEx = size(X.output[1])[end]
        return cat([reshape(lay, (:, nEx)) for lay in X.output]..., dims=1)
    else
        return cat(lay[:] for lay in X.output)
    end

end

Base.getindex(X::Scattered, i::Integer) = X.output[i+1]
function Base.getindex(X::ScatteredOut{T,N}, c::Colon,
    i::Union{<:AbstractArray,<:Integer}) where {T,N}
    ScatteredOut{T,N}(X.m, X.k, map(x -> x[axes(x)[1:end-1]..., i...], X.output))
end
function Base.getindex(X::ScatteredFull{T,N}, c::Colon,
    i::Union{<:AbstractArray,<:Integer}) where {T,N}
    ScatteredFull{T,N}(X.m, X.k, map(x -> x[axes(x)[1:end-1]..., i], X.data),
        map(x -> x[axes(x)[1:end-1]..., i], X.output))
end

function Base.getindex(X::Scattered, p::pathLocs{m}) where {m}
    ijk = p.indices
    λ(ii, access) = X[ii][access...]
    λ(ii, access::Nothing) = nothing
    λ(ii, access::AbstractArray) = X[ii][access]
    res = filter(x -> x != nothing, map(λ, (0:length(ijk)-1), ijk))
    if length(res) == 1
        return res[1]
    else
        return res
    end
end

# setting methods
function Base.setindex!(X::Scattered, v::Number, p::pathLocs{m}) where {m}
    res = X.output
    ijk = p.indices
    for (mm, ind) in enumerate(ijk)
        if typeof(ind) <: BitArray
            res[mm][ind] .= v
        elseif typeof(ind) <: Tuple{Vararg{Integer}}
            res[mm][ind...] = v
        elseif ind != nothing
            res[mm][ind...] .= v
        end
    end
end
function Base.setindex!(X::Scattered, v::Tuple, p::pathLocs{m}) where {m}
    res = X.output
    ijk = p.indices
    inputInd = 1
    for (mm, ind) in enumerate(ijk)
        if typeof(ind) <: Tuple{Vararg{Integer}}
            res[mm][ind...] = v[inputInd]
            inputInd += 1
        elseif typeof(ind) <: Union{BitArray,Array{Bool}}
            netSize = count(ind)
            res[mm][ind] = v[inputInd:inputInd+netSize-1]
            inputInd += netSize
        elseif ind != nothing
            res[mm][ind...] .= v[inputInd]
            inputInd += 1
        end
    end
end
function Base.setindex!(X::Scattered, v, p::pathLocs{m}) where {m}
    res = X.output
    ijk = p.indices
    inputInd = 1
    for (mm, ind) in enumerate(ijk)
        if typeof(ind) <: Tuple{Vararg{Integer}}
            res[mm][ind...] = v[inputInd]
            inputInd += 1
        elseif typeof(ind) <: Union{BitArray,Array{Bool}}
            netSize = count(ind)
            res[mm][ind] = v[inputInd:inputInd+netSize-1]
            inputInd += netSize
        elseif ind != nothing && size(v) != size(res[mm][ind...])
            netSize = prod(size(res[mm][ind...]))
            res[mm][ind...] = reshape(v[inputInd:inputInd+netSize-1], size(res[mm][ind...]))
            inputInd += 1
        elseif ind != nothing
            res[mm][ind...] = v
        end
    end
end


# can't normally union with a colon, which I think is just odd
import Base.union, Base.cat
union(a, b::Colon) = Colon()
union(a::Colon, b) = Colon()
union(a::Colon, b::Colon) = Colon()
# if we want to join several pathLocs
function cat(ps::pathLocs...)
    pairwise(a::Nothing, b::Nothing) = nothing
    pairwise(a::Nothing, b) = b
    pairwise(a, b::Nothing) = a
    pairwise(a, b) = union.(a, b)
    reduce((p1, p2) -> pathLocs(map(pairwise, p1.indices, p2.indices)...), ps)
end



"""
    nonZeroPaths(sc; wholePath=true, allTogetherInOne=false)
Given a `Scattered`, return the `pathLocs` where the `Scattered` is nonzero. `wholePath=true` if it returns the whole path, and not just the specific location in the signal. For example, if only `sc(pathLocs(1,(30,2)))` is nonzero, if `wholePath` is `true`, then `pathLocs(1,(2,))` will be returned while if `wholePath` is `false`, `pathLocs(1,(30,2))` will be returned instead.
if `allTogetherInOne` is `false`, then each location is returned separately, otherwise they are joined into a single `pathLocs`.
"""
function nonZeroPaths(sc; wholePath=true, allTogetherInOne=false)
    if wholePath
        return wholeNonzeroPaths(sc, allTogetherInOne)
    else
        return partNonzeroPaths(sc, allTogetherInOne)
    end
end


function partNonzeroPaths(sc, allTogetherInOne)
    if !allTogetherInOne
        error("wholePath false and each path separate not currently implemented because it makes the backend gross")
    end
    sz = size(sc)
    paths = Tuple{Vararg{pathLocs,0}}()
    Nd = ndims(sc)
    # zeroth layer
    nonZeroLocs = abs.(sc[0]) .> 0
    if any(nonZeroLocs)
        paths = (paths..., pathLocs(0, nonZeroLocs, d=Nd))
    end
    # first layer
    nonZero1 = map(i -> abs.(sc[pathLocs(1, i, d=Nd)]) .> 0, 1:sz[2][end-1])
    nonZero1 = map(x -> reshape(x, (size(x)[1:Nd]..., 1, size(x)[end])), nonZero1)
    nonZero1 = cat(nonZero1..., dims=Nd + 1)
    if any(nonZero1)
        paths = (paths..., pathLocs(1, nonZero1, d=Nd))
    end
    # second layer
    nonZero2 = abs.(sc[pathLocs(2, :)]) .> 0
    if any(nonZero2)
        paths = (paths..., pathLocs(2, nonZero2, d=Nd))
    end
    return cat(paths...)
end

function wholeNonzeroPaths(sc, allTogetherInOne=false)
    sz = size(sc)
    paths = Tuple{Vararg{pathLocs,0}}()
    # zeroth layer
    if maximum(sc[0] .> 0)
        paths = (paths..., pathLocs(0, :))
    end
    # first layer
    nonZero1 = filter(i -> maximum(abs.(sc[pathLocs(1, i)])) > 0, 1:sz[2][end-1])
    paths = (paths..., map(x -> pathLocs(1, x), nonZero1)...)

    # second layer
    nonZero2 = [pathLocs(2, (i, j)) for i = 1:sz[3][end-2], j = 1:sz[3][end-1] if maximum(abs.(sc[pathLocs(2, (i, j))])) > 0]
    paths = (paths..., nonZero2...)
    if allTogetherInOne
        return cat(paths...)
    else
        return paths
    end
end

"""
    addNextPath(addTo,addFrom)
    addNextPath(addFrom)

adds the next nonzero path in addFrom not present in addTo, ordered by layer, and
then the standard order for julia (dimension 1, then 2, 3,etc). If only handed `addFrom`
it makes a new pathLoc only containing the first non-empty path.

presently it only works for pathLocs whose indices are Boolean Arrays
"""
function addNextPath(addTo::pathLocs{m}, addFrom) where {m}
    shouldWeDo = foldl(checkShouldAdd, zip(addTo.indices, addFrom.indices), init = Tuple{}())
    inds = map(makeNext, addTo.indices, addFrom.indices, shouldWeDo)
    return pathLocs{m}(inds)
end
function checkShouldAdd(didPrevs, current)
    if !any(didPrevs) && current[1] != nothing
        addLoc = findfirst(current[1] .!= current[2])
        if addLoc != nothing
            return (didPrevs..., true)
        else
            return (didPrevs..., false)
        end
    else
        return (didPrevs..., false)
    end
end
function makeNext(current, fillWith, shouldDo)
    if shouldDo
        addLoc = findfirst(fillWith .!= current)
        newVersion = copy(current)
        newVersion[addLoc] = true
        return newVersion
    else
        return current
    end
end

addNextPath(addFrom::pathLocs{m}) where {m} = pathLocs{m}(foldl(makeSingle, addFrom.indices, init = Tuple{}()))

makeSingle(prev, x::Nothing) = (prev..., nothing)
makeSingle(x::Nothing) = (nothing,)
function makeSingle(x::BitArray)
    almostNull = falses(size(x))
    almostNull[findfirst(x)] = true
    return (prev..., almostNull)
end
function makeSingle(prev, x::BitArray)
    almostNull = falses(size(x))
    # only set this layer to true if all the previous are nothing
    if typeof(prev) <: Tuple{Vararg{Nothing}} # || !any.(any.(prev[prev .!=nothing]))
        almostNull[findfirst(x)] = true
    end
    return (prev..., almostNull)
end
