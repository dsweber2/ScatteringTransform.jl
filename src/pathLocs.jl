accessType = Union{Colon,<:Integer,<:AbstractArray{<:Union{Integer,Bool}},Nothing}
"""
    struct pathLocs{m}
        indices
    end
    pathLocs(varargs...; m::Int=2, d::Int=1, exs=Colon())
some words
    pathLocs(s::scattered)
make a pathLocs that accesses every location in the output
    pathLocs(ii, s::scattered)
make a pathLocs that accesses every location in layer `ii` of the output.
"""
struct pathLocs{m}
    indices
end
# specifying on
function pathLocs(varargs...; m::Int=2, d::Int=1, exs=Colon())
    if length(varargs) == 0
        # default constructor that includes everything
        paired = [(i, :) for i = 0:m]
    elseif length(varargs) % 2 == 0 && (varargs[1] isa Integer)
        # specifically enumerated, so alternates between a number and a reference shape
        # pair each layer with the tuple
        paired = [(varargs[i], varargs[i+1]) for i = 1:2:length(varargs)]
    elseif length(varargs) == m + 1
        # every layer specified
        paired = [(i, varargs[i+1]) for i = 0:m]
    end
    present = [x[1] for x in paired]
    notPresent = [ii for ii = 0:m if !(ii in present)] # unspecified layers
    paired = [paired..., [(x, nothing) for x in notPresent]...]
    paired = sort(paired, by=(x) -> x[1])
    indices = map(x -> parseOne(x, d, exs), paired)
    fullList = map(ii -> paired[ii+1], 0:m)
    return pathLocs{m + 1}((indices...,))
end

function parseOne(x, d, exs)
    lay, select = x
    if select == nothing
        return nothing
    end
    if lay == 0
        if select == Colon()
            return (map(x -> Colon(), 1:d)..., Colon(), exs)
        elseif length(select) == d + 2
            return select
        elseif typeof(select) <: AbstractArray && ndims(select) == d + 2
            return select
        elseif typeof(select) <: AbstractArray
            return (select, 1, exs)
        else
            return (select..., 1, exs)
        end
    elseif lay == 1
        if select == Colon()
            return (map(x -> Colon(), 1:d)..., Colon(), exs)
        elseif length(select) == d + 2
            return select
        elseif typeof(select) <: AbstractArray && ndims(select) == d + 2
            return select
        elseif typeof(select) <: AbstractArray || typeof(select) <: Number
            return (map(x -> Colon(), 1:d)..., select, exs)
        elseif length(select) == 1
            return (map(x -> Colon(), 1:d)..., select[1], exs)
        else
            return (select..., exs)
        end
    elseif lay >= 2
        if select == Colon()
            return (map(x -> Colon(), 1:d)..., Colon(), Colon(), exs)
        elseif (select isa Tuple) && length(select) >= d + 3
            return select
        elseif typeof(select) <: AbstractArray && ndims(select) == d + 3
            return select
        elseif typeof(select) <: AbstractArray # only specify the first layer
            return (map(x -> Colon(), 1:d+1)..., select, exs)
        elseif length(select) == 2 # specify both
            return (map(x -> Colon(), 1:d)..., select..., exs)
        else
            return (select..., 1, exs)
        end
    end
end



import Base: size
function size(p::pathLocs)
    ł(x::Nothing) = Tuple{Vararg{<:Integer,0}}()
    ł(x::AbstractArray) = size(x)
    # all that's left is a tuple of things
    ł(x) = map(a -> length(a), x)
    map(ł, p.indices)
end
