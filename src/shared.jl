abstract type scatteringTransform{Dimension, Depth} end
struct stFlux{Dimension, Depth, ChainType,D,E,F} <: scatteringTransform{Dimension, Depth}
    mainChain::ChainType
    normalize::Bool
    outputSizes::D
    outputPool::E
    settings::F
end

import Base.ndims
ndims(s::scatteringTransform{D}) where D = D
nPathDims(ii) = 1+max(min(ii-2,1),0) # the number of path dimensions at layer ii (zeroth
# is ii=1)
depth(s::scatteringTransform{Dim,Depth}) where {Dim, Depth} = Depth
function Base.show(io::IO, st::stFlux{Dim,Dep}) where {Dim,Dep}
    layers = st.mainChain.layers
    σ = layers[1].σ
    Nd = ndims(st)
    nFilters = [size(layers[i].weight,3)-1 for i=1:3:(3*Dim)]
    batchSize = getBatchSize(layers[1])
    print(io, "stFlux{$(Dep), Nd=$(Nd), filters=$(nFilters), σ = " *
          "$(σ), batchSize = $(batchSize), normalize = $(st.normalize)}")
end

# the type T is a type of frame transform that forms the backbone of the transform
# the type Dimension<:Integer gives the dimension of the transform

@doc """
        st(X::Array{T, N}, layers::stParallel, nonlinear::nl; fullOr::fullType=fullType(),# subsam::Sub = bspline(), thin::Bool=true, outputSubsample::Tuple{Int, Int}=(-1,-1), subsam::Bool=true, totalScales = [-1 for i=1:depth(layers)+1], percentage = .9, fftPlans = -1) where {T <: Real, S <: Union, N, nl <: Function, Sub <: resamplingMethod}
1D scattering transform using the stParallel layers. you can switch out the nonlinearity as well as the method of subsampling. Finally, the stType is a string. If it is "full", it will produce all paths. If it is "decreasing", it will only keep paths of increasing scale. If it is "collating", you must also include a vector of matrices.
# Arguments
- `nonlinear` : a type of nonlinearity. Should be a function that acts on Complex numbers
- `thin` : determines whether to wrap the output into a format that can be indexed using paths. `thin` cannot.
- `totalScales`, if positive, gives the number of non-averaging wavelets.
- `outputSubsample` is a tuple, with the first number indicating a rate, and the second number giving the minimum allowed size. If some of the entries are less than 1, it has different behaviour:
    + `(<1, x)` : subsample to x elements for each path.
    + `(<1, <1)` : no ssubsampling
    + `(x, <1)` : subsample at a rate of x, with at least one element kept in each path
- `fullOr::fullType=fullType()` : the structure of the transform either
       `fullType()`, `collatingType()` or `decreasingType()`. At the moment,
       only `fullType()` is functional.
- `fftPlans = false` if not `false`, it should be a 2D array of `Future`s, where the first index is the layer, and the second index the core. See `createFFTPlans` if you want to do this.
"""
struct stParallel{T, Dimension, Depth, subsampType, outType} <: scatteringTransform{Dimension, Depth}
    n::Tuple{Vararg{Int, Dimension}} # the dimensions of a single entry
    shears::Array{T,1} # the array of the transforms; the final of these is
    # used only for averaging, so it has length m+1
    subsampling::subsampType # for each layer, the rate of
    # subsampling. There is one of these for layer zero as well, since the
    # output is subsampled, so it should have length m+1
    outputSize::outType # a list of the size of a single output example
    # dimensions in each layer. The first index is layer, the second is
    # dimension (e.g. a 3 layer shattering transform is 3×2) TODO: currently
    # unused for the 1D case
end

function Base.show(io::IO, l::stParallel{T,D,Depth}) where {T,D,Depth}
    print(io, "stParallel{$T,$D} depth $(Depth), input size $(l.n), subsampling rates $(l.subsampling), outputsizes = $(l.outputSize)")
end

function eltypes(f::stParallel)
    return (eltype(f.shears), length(f.n))
end



"""
    listVargs = processArgs(m, varargs)
method to go from arguments given to the scattering transform constructor to 
those for the frame transform, e.g. shearlets or wavelet. `listVargs` is a list
of length `m` of one argument from each of vargs, with insufficiently long 
entries filled in by repeating the last value. An example:
```
julia> varargs
pairs(::NamedTuple) with 3 entries:
  :boundary      => PerBoundary()
  :frameBound    => [1, 1]
  :normalization => (Inf, Inf)

julia> listVargs = processArgs(3,varargs)
(Base.Iterators.Pairs{Symbol,Any,Tuple{Symbol,Symbol,Symbol},NamedTuple{(:boundary, :frameBound, :normalization),Tuple{PerBoundary,Int64,Float64}}}(:boundary => PerBoundary(),:frameBound => 1,:normalization => Inf), Base.Iterators.Pairs{Symbol,Any,Tuple{Symbol,Symbol,Symbol},NamedTuple{(:boundary, :frameBound, :normalization),Tuple{PerBoundary,Int64,Float64}}}(:boundary => PerBoundary(),:frameBound => 1,:normalization => Inf), Base.Iterators.Pairs{Symbol,Any,Tuple{Symbol,Symbol,Symbol},NamedTuple{(:boundary, :frameBound, :normalization),Tuple{PerBoundary,Int64,Float64}}}(:boundary => PerBoundary(),:frameBound => 1,:normalization => Inf))

julia> listVargs[1]
pairs(::NamedTuple) with 3 entries:
  :boundary      => PerBoundary()
  :frameBound    => 1
  :normalization => Inf

julia> listVargs[2]
pairs(::NamedTuple) with 3 entries:
  :boundary      => PerBoundary()
  :frameBound    => 1
  :normalization => Inf

julia> listVargs[3]
pairs(::NamedTuple) with 3 entries:
  :boundary      => PerBoundary()
  :frameBound    => 1
  :normalization => Inf
```
"""
function processArgs(m, varargs)
    keysVarg = keys(varargs)
    valVarg = map(v->makeTuple(m,v), values(varargs))
    pairedArgs = Iterators.Pairs(valVarg,keysVarg)
    listVargs = ntuple(i -> Iterators.Pairs(map(x->x[i],valVarg), keysVarg), m)
end

function makeTuple(m, v::Tuple)
    if length(v) >= m
        return v
    end
    return (v...,ntuple(i->v[end], m-length(v))...)
end
makeTuple(m, v::AbstractArray) = makeTuple(m, (v...,))
makeTuple(m, v) = ntuple(i->v, m)









# shouldn't be necessary
# function cwtDefaults(varargs)
#     keyLimes = keys(varargs)
#     DefaultList = ((:averagingLength,4),
#                    (:averagingType,ContinuousWavelets.Father()),
#                    (:boundary,DEFAULT_BOUNDARY),
#                    (:frameBound,1),
#                    (:normalization,Inf),
#                    (:decreasing, 4.0))
#     for (key,val) in DefaultList
        
#     end
# end
