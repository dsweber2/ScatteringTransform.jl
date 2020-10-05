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
