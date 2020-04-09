# TODO: make a version of conv that is efficient for small sizes
############################### TRANSFORM TYPES ###############################

# TODO: resizingrates
# the type T is a type of frame transform that forms the backbone of the transform
# the type D<:Integer gives the dimension of the transform
struct layeredTransform{T, D}
    m::Int # the number of layers, not counting the zeroth layer
    n::Tuple{Vararg{Int,D}} # the dimensions of a single entry
    shears::Array{T,1} # the array of the transforms; the final of these is
    # used only for averaging, so it has length m+1
    subsampling::Array{Float32, 1} # for each layer, the rate of
    # subsampling. There is one of these for layer zero as well, since the
    # output is subsampled, so it should have length m+1
    outputSize::Array{Int, 2} # a list of the size of a single output example
    # dimensions in each layer. The first index is layer, the second is
    # dimension (e.g. a 3 layer shattering transform is 3×2) TODO: currently
    # unused for the 1D case
    function layeredTransform{T,D}(m::Int, n::Tuple{Vararg{Int, D}},
                                   shears::Array{T,1},
                                   subsampling::Array{Float32, 1},
                                   outputSize::Array{Int, 2}) where {T,D}
        @assert (D==1 && T<:CFW)
        return new(m, n, shears, subsampling, outputSize)
    end
end

function Base.show(io::IO, l::layeredTransform{T,D}) where {T,D}
    print(io, "layeredTransform{$T,$D} depth $(l.m), input size $(l.n), subsampling rates $(l.subsampling), outputsizes = $(l.outputSize)")
end


function eltypes(f::layeredTransform{T, D}) where {T, D}
    return (eltype(f.shears), length(f.n))
end
################################################################################
######################## Continuous Wavelet Transform  #########################
################################################################################

# the fully specified version
# TODO: use a layered transform parameter to determine if we're returning a st or thinst instead
@doc """
    layers = layeredTransform(m::S, Xlength::S; nScales::Array{S,1} = [8 for
                                                                   i=1:(m+1)],
                          subsampling::Array{T,1} = [2 for i=1:(m+1)],
                          CWTType::WT.ContinuousWaveletClass=WT.morl,
                          averagingLength::Array{S,1} = [S(4) for i=1:(m+1)],
                          averagingType::Array{<:WT.Average,1} = [WT.Father() for
                                                                   i=1:(m+1)],
                          boundary::Array{W,1} = [WT.DEFAULT_BOUNDARY for
                                                  i=1:(m+1)],
                          frameBounds=[1 for i=1:(m+1)]) where {S <: Integer,
                                                     T <: Real,
                                                     W <: WT.WaveletBoundary}
            layeredTransform(m::S, Xlength::S, nScales::Array{T,1},
                                  subsampling::Array{T,1},
                                  CWTType::WT.ContinuousWaveletClass,
                                  averagingLength::Array{S,1} = ceil.(S,nScales*2),
                                  averagingType::Array{Symbol,1} = [Father() for
                                                                    i=1:(m+1)],
                                  boundary::Array{W,1} = [WT.DEFAULT_BOUNDARY for
                                                          i=1:(m+1)]) where {S <:
                                                                             Integer, 
                                                                             T <: Real,
                                                                             W <: WT.WaveletBoundary}
whether the second argument is a single number or a tuple determines whether
you get a 1D wavelet transform
 """
function layeredTransform(m::S, Xlength::S, nScales, subsampling,
                          CWTType::Array{<:WT.ContinuousWaveletClass,1},
                          averagingLength = ceil.(S,4),
                          averagingType::Array{<:WT.Average,1} = [WT.Father() for
                                                            i=1:(m+1)],
                          boundary::Array{<:WT.WaveletBoundary,1} =
                          [WT.DEFAULT_BOUNDARY for i=1:(m+1)],
                          frameBounds=[1 for i=1:(m+1)], normalization = [Inf
                                                                          for
                                                                          i=1:(m+1)],
                          decreasing = [4.0 for i=1:(m+1)]) where {S <: Integer, 
                                                                 T <: Real}
    @assert m+1 == size(subsampling, 1)
    @assert m+1 == size(nScales, 1)
    
    @info "Treating as a 1D Signal. Vector Lengths: $Xlength nScales:" *
            "$nScales subsampling: $subsampling"
    shears = [wavelet(CWTType[i]; s = nScales[i],
                      boundary = boundary[i], 
                      averagingType = averagingType[i],
                      averagingLength = averagingLength[i],
                      frameBound = frameBounds[i],
                      normalization = normalization[i],
                      decreasing = decreasing[i]) for (i,x) in
              enumerate(subsampling)]
    
    layeredTransform{typeof(shears[1]), 1}(m, (Xlength,), shears,
                                           Float32.(subsampling), [1 1])
end




# version with explicit name calling
function layeredTransform(m::S, Xlength::S; nScales = [8.0 for i=1:(m+1)],
                          subsampling = [2 for i=1:(m+1)],
                          CWTType=WT.morl,
                          averagingLength = [4 for i=1:(m+1)],
                          averagingType::Array{<:WT.Average,1} = [WT.Father() for
                                                                   i=1:(m+1)],
                          boundary::Array{W,1} = [WT.DEFAULT_BOUNDARY for
                                                  i=1:(m+1)],
                          frameBounds=[1 for i=1:(m+1)], 
                          normalization = [Inf for i=1:(m+1)],
                          decreasing = [4.0 for i=1:(m+1)]) where {S <: Integer,
                                                     T <: Real,
                                                     W <: WT.WaveletBoundary}
    if typeof(CWTType) <: WT.ContinuousWaveletClass
        CWTType = [CWTType for i=1:m+1]
    end
    layeredTransform(m, Xlength, nScales, subsampling, CWTType,
                     averagingLength, averagingType, boundary, frameBounds, normalization, decreasing)
end



#################################################################################
########################### scattered type ######################################
#################################################################################

# TODO: tests for the data collators and the subsampling
# TODO: Write a version of this that accomodates things that are too big to
# hold in memory
# TODO: write a version of this that uses sparse matrices
# TODO: adapt for complex input and/or complex wavelets
struct scattered{T,N}
    m::Int64 # number of layers, counting the zeroth layer
    k::Int64 # the meta-dimension of the signals (should be either 1 or 2)
    data::Array{Array{T, N}, 1} #original last dimension is time, new
                                         #path/frequency
    output::Array{Array{T, N}, 1} # The final averaged results; this is the
                                # output from the entire system 
    function scattered{T,N}(m, k, data, output) where {T<:Number, N}
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

function Base.show(io::IO, sc::scattered{T,D}) where {T,D}
    print(io, "scattered{$T,$D} depth $(sc.m)")
end


import Base:getindex
Base.getindex(X::scattered, i::Union{AbstractArray, <:Integer}) = X.output[i]

function Base.getindex(X::scattered, p::pathType)
    # if it's not an integer type, then the last index is the only weird one
    k = X.k
    from = X.output[p.m+1]
    ax = axes(from)

    lastInd = p.Idxs[end]
    if typeof(lastInd) <: Integer
        (relIndex, nShears) = relativeIndex(X.output, k, p, p.Idxs[1:end-1])
    else
        (relIndex, nShears) = relativeIndex(X.output, k, p, p.Idxs[1:end-1])
    end
    if typeof(lastInd) <: Colon
        relIndex = relIndex .+ (1:nShears[end-1])
    else
        relIndex =  relIndex .+ (lastInd)
    end
    return from[ax[1:k]..., relIndex, ax[k+2:end]...]
end
function relativeIndex(output, k, p, idxs)
        nShears = [size(x,k+1) for x in output[1:p.m+1]]
        nShears = [floor(Int, x/prod(nShears[1:i-1])) for (i,x) in enumerate(nShears)]
        nShears = [nShears... 1]
    if length(idxs)> 0
        relIndex = sum([(px-1)*prod(nShears[i+2:end]) for (i,px) in enumerate(idxs)])
    else
        relIndex = 0
    end
    return (relIndex, nShears)
end

@doc """
      scattered(m, k, fixDim::Array{<:Real, 1}, n::Array{<:Real, 2}, q::Array{<:Real, 1}, T::DataType)

  element type is T and N the total number of indices, including both signal dimension and any example indices. k is the actual dimension of a single signal.
  """
function scattered(m, k, fixDim::Array{<:Real, 1}, n::Array{<:Real, 2},
                   q::Array{<:Real, 1}, T::DataType)
    @assert m+1==size(n,1)
    @assert m+1==length(q)
    @assert   k==size(n,2)
    fixDim = Int.(fixDim)
    n = Int.(n)
    q = Int.(q)
    N = k + length(fixDim)+1
    data = [zeros(T, n[i,:]..., prod(q[1:i].-1), fixDim...) for
            i=1:m+1]
    output = [zeros(T,  n[i,:]..., prod(q[1:(i-1)].-1), fixDim...) for i=1:m+1]
    return scattered{T,N}(m, k, data, output)
end


# TODO: check performance tips to see if I should add some way of discerning the type of N
function scattered(layers::layeredTransform{S,1}, X::Array{T,N};
                   totalScales=[-1 for i=1:layers.m +1], outputSubsample=(-1,-1)) where {T <: Real, N,
                                                                S}
    if N == 1
        X = reshape(X, (size(X)..., 1));
    end

    n, q, dataSizes, outputSizes, resultingSize = 
        calculateSizes(layers, outputSubsample, size(X),totalScales =
                       totalScales)

    if 1==N
        zerr = [zeros(T, n[i], q[i]) for i=1:layers.m+1]
        output = [zeros(T, resultingSize[i], q[i]) for i=1:layers.m+1]
    else
        zerr=[zeros(T, n[i], q[i], size(X)[2:end]...) for
              i=1:layers.m+1]
        output = [zeros(T, resultingSize[i], q[i], size(X)[2:end]...)
                  for i=1:layers.m+1]
        @info "" [size(x) for x in output]
    end
    zerr[1][:, 1, axes(X)[2:end]...] = copy(X)
    return scattered{T,N+1}(layers.m, 1, zerr, output)
end
