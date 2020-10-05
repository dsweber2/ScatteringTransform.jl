abstract type Scattered{T,N} end


# TODO: tests for the data collators and the subsampling
# TODO: Write a version of this that accomodates things that are too big to
# hold in memory
# TODO: write a version of this that uses sparse matrices
# TODO: adapt for complex input and/or complex wavelets
struct ScatteredFull{T,N} <:Scattered{T,N}
    m::Int64 # number of layers, counting the zeroth layer
    k::Int64 # the meta-dimension of the signals (should be either 1 or 2)
    data::Array{AbstractArray{T, N}, 1} #original last dimension is time, new
                                         #path/frequency
    output::Array{AbstractArray{T, N}, 1} # The final averaged results; this is the
                                # output from the entire system 
    function ScatteredFull{T,N}(m, k, data, output) where {T<:Number, N}
        @assert N > k # The size of the arrays must be at least 1 larger than the
                    # input dimension
        m = Int64(m)
        k = Int64(k)
        # check data
        singlentry = eltype(data[1])
        arrayType = eltype(data)
        @assert arrayType <: AbstractArray{singlentry, N}
        @assert size(data,1)==m+1

        # check output
        singlentry = eltype(output[1])
        arrayType = eltype(output)
        @assert arrayType <: AbstractArray{singlentry, N}
        @assert size(output,1)==m+1
        new(m, k, data, output)
    end
end

"""
A simple wrapper for the results of the scattering transform. Its one field
`result` contains a tuple of the results from each layer, in the order zero,
one, two. Say we have an example `s`. You can access layer i by `s[i]`, so
`s[0]` gives the zeroth layer
"""
struct ScatteredOut{T,N} <:Scattered{T,N}
    m::Int# number of layers, counting the zeroth layer
    k::Int# the meta-dimension of the signals (should be either 1 or 2)
    output
end
ScatteredOut(output; k=1) = ScatteredOut{eltype(output),
                                         ndims(output[1])}(length(output)-1, k,
                                                           output)

import Base.+, Base.-, Base.broadcast, Base.length, Base.size, Base.iterate, Base.broadcastable
function +(a::ScatteredOut, b::ScatteredOut)
    @assert a.m== b.m
    @assert a.k== b.k
    ScatteredOut(a.m,a.k,map((x,y)->x+y, a.output, b.output))
end
function +(a::ScatteredFull, b::ScatteredFull) 
    @assert a.m== b.m
    @assert a.k== b.k
    ScatteredFull(a.m, a.k, map((x,y)->x+y, a.data, b.data), map((x,y)->x+y, a.output, b.output))
end
function -(a::ScatteredOut, b::ScatteredOut) 
    @assert a.m== b.m
    @assert a.k== b.k
    ScatteredOut(a.m,a.k,map((x,y)->x-y, a.output, b.output))
end
function -(a::ScatteredFull, b::ScatteredFull) 
    @assert a.m== b.m
    @assert a.k== b.k
    ScatteredFull(a.m,a.k,map((x,y)->x-y, a.data, b.data), map((x,y)->x-y, a.output, b.output))
end

broadcast(f, a::ScatteredOut, varargs...) =
    ScatteredOut(a.m, a.k, map(x->broadcast(f,x,varargs...), a.output))
broadcast(f, a::ScatteredFull, varargs...) =
    ScatteredFull(a.m, a.k, map(x->broadcast(f,x,varargs...), a.data),
                  map(x->broadcast(f,x,varargs...), a.output))
broadcastable(a::Scattered) = a.output
iterate(a::Scattered) = iterate(a.output)
iterate(a::Scattered, b) = iterate(a.output, b)
length(sct::Scattered) = length(sct.output)-1
ndims(sct::Scattered) = sct.k
size(sct::ScatteredOut) = map(x->size(x), sct.output)
size(sct::ScatteredFull) = map(x->size(x), sct.data)
import Base.similar
similar(sct::ScatteredOut) = ScatteredOut(sct.m, sct.k,
                                       map(x->similar(x), sct.output))
similar(sct::ScatteredFull) = ScatteredFull(sct.m, sct.k,
                                        map(x->similar(x), sct.data),
                                        map(x->similar(x), sct.output))
import Statistics.mean
function mean(a::ScatteredOut)
    ScatteredOut(a.m,a.k,map(x->mean(x,dims=ndims(x)), a.output))
end
import Base.cat

function cat(sc::ScatteredOut, sc1::ScatteredOut, sc2::Vararg{<:ScatteredOut,
                                                              N}) where N
    @assert !any([sc.m != sc1.m, (sc.m .!= map(s->s.m, sc2))])# check they're
    @assert !any([sc.k != sc1.k, (sc.k .!= map(s->s.k, sc2))])# all equal
    outputs = map(x->x.output, (sc, sc1, sc2...))
    output = map(x->cat(x...,dims=ndims(x[1])), zip(outputs...))
    return ScatteredOut(sc.m, sc.k, tuple(output...))
end
function cat(sc::ScatteredFull, sc1::ScatteredFull, sc2::Vararg{<:ScatteredFull,
                                                              N}) where N
    @assert !any([sc.m != sc1.m, (sc.m .!= map(s->s.m, sc2))])# check they're
    @assert !any([sc.k != sc1.k, (sc.k .!= map(s->s.k, sc2))])# all equal
    outputs = map(x->x.output, (sc, sc1, sc2...))
    output = map(x->cat(x...,dims=ndims(x[1])), zip(outputs...))
    datas = map(x->x.data, (sc, sc1, sc2...))
    data = map(x->cat(x..., dims=ndims(x[1])), zip(datas...))
    return ScatteredFull(sc.m, sc.k, tuple(data...), tuple(output...))
end


function Base.show(io::IO, s::ScatteredOut{T,N}) where {T,N}
    print(io, "ScatteredOut{T,N} $(s.k) dim. OutputSizes:")
    outputSizes = map(x->size(x), s.output)
    for x in outputSizes
        print(io,"\n    $(x)")
    end
end

function Base.show(io::IO, s::ScatteredFull{T,N}) where {T,N}
    print(io, "ScatteredFull{T,N} $(s.k) dim. Data sizes:")
    dataSizes = map(x->size(x), s.data)
    for x in dataSizes
        print(io,"\n    $(x)")
    end
    print(io,"\nOutputSizes:")
    outputSizes = map(x->size(x), s.output)
    for x in outputSizes
        print(io,"\n    $(x)")
    end
end

# pathLocs constructors that use a Scattered type
function pathLocs(s::Scattered)
    return pathLocs{length(s.output)}(map(x->axes(x), s.output))
end

function pathLocs(ii, s::Scattered)
    return pathLocs{length(s.output)}(map(x->axes(x), s.output[ii]))
end

# access methods
import Base:getindex
Base.getindex(X::Scattered, i::AbstractArray) = X.output[i .+ 1]
Base.getindex(X::Scattered, i::Integer) = X.output[i+1]
function Base.getindex(X::ScatteredOut, c::Colon, 
                       i::Union{<:AbstractArray, <:Integer})
    ScatteredOut(X.m, X.k, map(x-> x[axes(x)[1:end-1]..., i], X.output))
end
function Base.getindex(X::ScatteredFull, c::Colon, 
                       i::Union{<:AbstractArray, <:Integer})
    ScatteredFull(X.m, X.k, map(x-> x[axes(x)[1:end-1]..., i], X.data),
                  map(x-> x[axes(x)[1:end-1]..., i], X.output))
end

function Base.getindex(X::Scattered, p::pathLocs{m}) where m
    ijk = p.indices
    λ(ii, access) = X[ii][access...]
    λ(ii, access::Nothing) = nothing
    λ(ii, access::AbstractArray) = X[ii][access]
    res = filter(x->x!=nothing, map(λ, (0:length(ijk)-1), ijk))
    if length(res) == 1
        return res[1]
    else
        return res
    end
end

# setting methods
function Base.setindex!(X::Scattered, v::Number, p::pathLocs{m}) where m
    res = X.output
    ijk = p.indices
    for (mm,ind) in enumerate(ijk)
        if typeof(ind) <: BitArray
            res[mm][ind] .= v
        elseif typeof(ind) <: Tuple{Vararg{<:Integer}}
            res[mm][ind...] = v
        elseif ind!=nothing
            res[mm][ind...] .= v
        end
    end
end
function Base.setindex!(X::Scattered, v, p::pathLocs{m}) where m
    res = X.output
    ijk = p.indices
    inputInd = 1
    for (mm,ind) in enumerate(ijk)
        if typeof(ind) <: Tuple{Vararg{<:Integer}}
            res[mm][ind...] = v[inputInd]
            inputInd+=1
        elseif typeof(ind) <:Union{BitArray, Array{Bool}}
            netSize = count(ind)
            res[mm][ind] = v[inputInd:inputInd+netSize-1]
            inputInd+=netSize
        elseif  ind!=nothing
            res[mm][ind...] .= v[inputInd]
            inputInd+=1
        end
    end
end


# can't normally union with a colon, which I think is just odd
import Base.union, Base.cat
union(a, b::Colon) = Colon()
union(a::Colon, b) = Colon()
union(a::Colon, b::Colon) = Colon()
# if we want to join several pathLocs
function cat(ps::Vararg{pathLocs, N}) where N
    ł(a::Nothing,b::Nothing)=nothing
    ł(a::Nothing,b) = b
    ł(a,b::Nothing) = a
    ł(a,b) = union.(a,b)
    reduce((p1,p2)->pathLocs(map(ł, p1.indices, p2.indices)), ps)
end



"""
    paths = nonZeroPaths(sc; wholePath=true, allTogetherInOne=false)
given a Scattered, return the pathLocs where the Scattered is nonzero. `wholePath` is true if it returns the whole path, and not just the specific location in the signal. For example, if only `sc(pathLocs(1,(30,2)))` is nonzero, if `wholePath` is true, then `pathLocs(1,(2,))` will be returned while if it is false, `pathLocs(1,(30,2))` will be returned instead.
if `allTogetherInOne` is false, then each location is returned separately, otherwise they are joined into a single `pathLocs`.
"""
function nonZeroPaths(sc; wholePath=true, allTogetherInOne=false)
    if wholePath
        return wholeNonzeroPaths(sc,allTogetherInOne)
    else
        return partNonzeroPaths(sc,allTogetherInOne)
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
    typeof(nonZeroLocs)
    if any(nonZeroLocs)
        paths = (paths..., pathLocs(0, nonZeroLocs,d=Nd))
    end
    # first layer
    nonZero1 = map(i->abs.(sc[pathLocs(1,i,d=Nd)]) .>0, 1:sz[2][end-1])
    nonZero1 = map(x->reshape(x, (size(x)[1:Nd]..., 1, size(x)[end])), nonZero1)
    nonZero1 = cat(nonZero1..., dims = Nd+1)
    if any(nonZero1)
        paths = (paths..., pathLocs(1,nonZero1, d = Nd))
    end
    # second layer
    nonZero2 = abs.(sc[pathLocs(2,:)]) .>0
    if any(nonZero2)
        paths = (paths..., pathLocs(2,nonZero2, d=Nd))
    end
    return cat(paths...)
end

function wholeNonzeroPaths(sc, allTogetherInOne=false)
    sz = size(sc)
    paths = Tuple{Vararg{pathLocs,0}}()
    # zeroth layer
    if maximum(sc[0] .>0)
        paths = (paths..., pathLocs(0, :))
    end
    # first layer
    nonZero1 = filter(i->maximum(abs.(sc[pathLocs(1,i)]))>0, 1:sz[2][end-1])
    paths = (paths..., map(x->pathLocs(1, x), nonZero1)...)
    
    # second layer
    nonZero2 = [pathLocs(2, (i,j)) for i=1:sz[3][end-2], j=1:sz[3][end-1] if maximum(abs.(sc[pathLocs(2,(i,j))]))>0]
    paths = (paths..., nonZero2...)
    if allTogetherInOne
        return cat(paths...)
    else
        return paths
    end
end

