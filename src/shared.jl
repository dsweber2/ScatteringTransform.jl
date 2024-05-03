@doc """
     scatteringTransform{Dimension,Depth}
 The abstract type and constructor for scattering transforms. The specific types are `stFlux` in this package, and `stParallel` in [ParallelScattering.jl](https://github.com/dsweber2/ParallelScattering.jl/).
 """
abstract type scatteringTransform{Dimension,Depth} end

struct stFlux{Dimension,Depth,ChainType,D,E,F} <: scatteringTransform{Dimension,Depth}
    mainChain::ChainType
    normalize::Bool
    outputSizes::D
    outputPool::E
    settings::F
end

import Base.ndims
@doc """
     ndims(s::scatteringTransform{D})
 given a scattering transform `s`, return the number of layers `Depth`.
 """
ndims(s::scatteringTransform{D}) where {D} = D
nPathDims(ii) = 1 + max(min(ii - 2, 1), 0) # the number of path dimensions at layer ii (zeroth
# is ii=1)
@doc """
     depth(s::scatteringTransform{Dim,Depth})
 given a scattering transform, return the number of layers `Depth`.
 """
depth(s::scatteringTransform{Dim,Depth}) where {Dim,Depth} = Depth

function Base.show(io::IO, st::stFlux{Dim,Dep}) where {Dim,Dep}
    layers = st.mainChain.layers
    σ = st.settings[:σ]
    Nd = ndims(st)
    nFilters = [length(layers[i].weight) - 1 for i = 1:3:(3*Dep)]
    batchSize = getBatchSize(layers[1])
    print(io, "stFlux{Nd=$(Nd), m=$(Dep), filters=$(nFilters), σ = " *
              "$(σ), batchSize = $(batchSize), normalize = $(st.normalize)}")
end

# the type T is a type of frame transform that forms the backbone of the transform
# the type Dimension<:Integer gives the dimension of the transform

"""
    processArgs(m, varargs) -> listVargs
Go from arguments given to the scattering transform constructor to those for the wavelet or frame transform. `listVargs` is a list of length `m` of one argument from each of vargs, with insufficiently long entries filled in by repeating the last value. For a list of these arguments, see the documentation for [`stFlux`](@ref).

# Examples
```jldoctest
julia> using ContinuousWavelets, ScatteringTransform

julia> varargs = ( :boundary      => PerBoundary(), :frameBound    => [1, 1], :normalization => (Inf, Inf))
(:boundary => PerBoundary(), :frameBound => [1, 1], :normalization => (Inf, Inf))

julia> varargs
(:boundary => PerBoundary(), :frameBound => [1, 1], :normalization => (Inf, Inf))

julia> listVargs = ScatteringTransform.processArgs(3,varargs)
(Base.Pairs{Int64, Pair{Symbol}, Base.OneTo{Int64}, Tuple{Pair{Symbol, ContinuousWavelets.PerBoundary}, Pair{Symbol, Vector{Int64}}, Pair{Symbol, Tuple{Float64, Float64}}}}(1 => (:boundary => PerBoundary()), 2 => (:frameBound => [1, 1]), 3 => (:normalization => (Inf, Inf))), Base.Pairs{Int64, Pair{Symbol}, Base.OneTo{Int64}, Tuple{Pair{Symbol, ContinuousWavelets.PerBoundary}, Pair{Symbol, Vector{Int64}}, Pair{Symbol, Tuple{Float64, Float64}}}}(1 => (:boundary => PerBoundary()), 2 => (:frameBound => [1, 1]), 3 => (:normalization => (Inf, Inf))), Base.Pairs{Int64, Pair{Symbol}, Base.OneTo{Int64}, Tuple{Pair{Symbol, ContinuousWavelets.PerBoundary}, Pair{Symbol, Vector{Int64}}, Pair{Symbol, Tuple{Float64, Float64}}}}(1 => (:boundary => PerBoundary()), 2 => (:frameBound => [1, 1]), 3 => (:normalization => (Inf, Inf))))

julia> listVargs[1]
pairs(::Tuple{Pair{Symbol, ContinuousWavelets.PerBoundary}, Pair{Symbol, Vector{Int64}}, Pair{Symbol, Tuple{Float64, Float64}}}) with 3 entries:
  1 => :boundary=>PerBoundary()
  2 => :frameBound=>[1, 1]
  3 => :normalization=>(Inf, Inf)

julia> listVargs[2]
pairs(::Tuple{Pair{Symbol, ContinuousWavelets.PerBoundary}, Pair{Symbol, Vector{Int64}}, Pair{Symbol, Tuple{Float64, Float64}}}) with 3 entries:
  1 => :boundary=>PerBoundary()
  2 => :frameBound=>[1, 1]
  3 => :normalization=>(Inf, Inf)

julia> listVargs[3]
pairs(::Tuple{Pair{Symbol, ContinuousWavelets.PerBoundary}, Pair{Symbol, Vector{Int64}}, Pair{Symbol, Tuple{Float64, Float64}}}) with 3 entries:
  1 => :boundary=>PerBoundary()
  2 => :frameBound=>[1, 1]
  3 => :normalization=>(Inf, Inf)

```
"""
function processArgs(m, varargs)
    keysVarg = keys(varargs)
    valVarg = map(v -> makeTuple(m, v), values(varargs))
    pairedArgs = Iterators.Pairs(valVarg, keysVarg)
    listVargs = ntuple(i -> Iterators.Pairs(map(x -> x[i], valVarg), keysVarg), m)
end

function makeTuple(m, v::Tuple)
    if length(v) >= m
        return v
    end
    return (v..., ntuple(i -> v[end], m - length(v))...)
end
makeTuple(m, v::AbstractArray) = makeTuple(m, (v...,))
makeTuple(m, v) = ntuple(i -> v, m)
