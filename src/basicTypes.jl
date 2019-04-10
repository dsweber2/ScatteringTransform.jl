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
        @assert (D==2 && T<:Shearletsystem2D)  || (D==1 && T<:CFWA)
        return new(m, n, shears, subsampling, outputSize)
    end
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
        layers = layeredTransform(m::S, Xlength::S, nScales::Array{T,1},
                                  subsampling::Array{T,1},
                                  CWTType::WT.ContinuousWaveletClass,
                                  averagingLength::Array{S,1} = ceil.(S,nScales*2),
                                  averagingType::Array{Symbol,1} = [:Mother for
                                                                    i=1:(m+1)],
                                  boundary::Array{W,1} = [WT.DEFAULT_BOUNDARY for
                                                          i=1:(m+1)]) where {S <:
                                                                             Integer, 
                                                                             T <: Real,
                                                                             W <: WT.WaveletBoundary}
      layers = layeredTransform(m::S, Xsizes::Tuple{<:Integer,<:Integer},
                                nScales::Array{T,1}, subsampling::Array{T,1},
                                shearlets<:W,
                                shearLevels::Array{<:Array{<:Integer,1}},
                                gpu::Bool=false, percentage,
                                typeBecomes::DataType=Float32) where {S<:Integer,
                                                                      T<:Real,
                                                                      W<:Shearlab.Shearletsystem2D}
whether the second argument is a single number or a tuple determines whether
you get a 1D wavelet transform or 2D shearlet transform
 """
function layeredTransform(m::S, Xlength::S, nScales::Array{S,1},
                          subsampling::Array{T,1},
                          CWTType::V,
                          averagingLength::Array{S,1} = ceil.(S,nScales*2),
                          averagingType::Array{Symbol,1} = [:Mother for
                                                            i=1:(m+1)],
                          boundary::Array{W,1} = [WT.DEFAULT_BOUNDARY for
                                                  i=1:(m+1)]) where {S <:
                                                                     Integer, 
                                                                     T <: Real,
                                                                     W <:
                                                                     WT.WaveletBoundary,
                                                                     V <:
                                                                     WT.ContinuousWaveletClass}
    @assert m+1==size(subsampling,1)
    @assert m+1==size(nScales,1)
    
    println("Treating as a 1D Signal. Vector Lengths: $Xlength nScales:" *
            "$nScales subsampling: $subsampling")
    shears = [wavelet(CWTType, nScales[i], averagingLength[i],
                      averagingType[i], boundary[i]) for (i,x) in
              enumerate(subsampling)]
    layeredTransform{typeof(shears[1]), 1}(m, (Xlength,), shears,
                                           Float32.(subsampling), [1 1])
end


################################################################################
######################### Shearlet Transform  ##################################
################################################################################
function layeredTransform(m::S, Xsizes::Tuple{<:Integer, <:Integer},
                          nScales::Array{<:Integer, 1}, subsampling::Array{T, 1},
                          shearLevels::Array{<:Array{<:Integer, 1}},
                          padding::Bool, gpu::Bool=false, percentage=.9,
                          typeBecomes::DataType=Float32) where {S<:Integer,
                                                                T<:Real,
                                                                W<:Shearlab.Shearletsystem2D}
    @assert m+1==size(subsampling, 1)
    @assert m+1==size(nScales, 1)

    rows = sizes(bsplineType(), subsampling, Xsizes[1])[1:end-1]
    cols = sizes(bsplineType(), subsampling, Xsizes[2])[1:end-1]
    
    println("Treating as a 2D signal. Vector Lengths: $Xsizes nScales:" *
            "$nScales subsampling: $subsampling")
    shears =  [Shearlab.getshearletsystem2D(rows[i], cols[i], nScales[i],
                                            shearLevels[i];
                                            typeBecomes=typeBecomes) for (i,x) in
               enumerate(subsampling)]
    outputSizes = getResizingRates(shears, length(shears)-1, percentage=percentage)
    # adjust the shearlets to handle the padding
    for (i, shear) in enumerate(shears)
        padBy = getPadBy(shears[i])
        shearlets = real.(fftshift(ifft(ifftshift(shear.shearlets, (1, 2)), (1, 2)), (1, 2)))
        newShears = zeros(eltype(shearlets), (size(shear.shearlets)[1:2]
                                              .+ 2 .* padBy)...,
                          size(shear.shearlets, 3))
        if typeBecomes<:Real
            P = plan_rfft(view(newShears, :, :, 1))
            newShears = zeros(Complex{eltype(shearlets)}, div(size(newShears,1), 2)+1,
                              size(newShears)[2:3]...)
        else
            P = plan_fft(view(newShears, :, :, 1))
            newShears = zeros(eltype(shearlets), div(size(newShears,1), 2)+1,
                              size(newShears)[2:3]...)
        end
        for j=1:size(shear.shearlets, 3)
            newShears[:, :, j] = fftshift(P * pad(shearlets[:,:,j],
                                                  padBy))
        end
        shears[i] = Shearletsystem2D{typeBecomes}(newShears, shear.size,
                                                  shear.shearLevels,
                                                  shear.full, shear.nShearlets,
                                                  shear.shearletIdxs,
                                                  shear.dualFrameWeights,
                                                  shear.RMS, shear.isComplex,
                                                  shear.gpu, shear.support)
    end
    return layeredTransform{typeof(shears[1]), 2}(m, Xsizes, shears,
                                                  Float32.(subsampling), outputSizes)
end


# versions with explicit name calling
function layeredTransform(m::S, Xlength::S; nScales::Array{S,1} = [8 for
                                                                   i=1:(m+1)],
                          subsampling::Array{T,1} = [2 for i=1:(m+1)],
                          CWTType::WT.ContinuousWaveletClass=WT.morl,
                          averagingLength::Array{S,1} = ceil.(S, nScales/2),
                          averagingType::Array{Symbol,1} = [:Mother for
                                                                   i=1:(m+1)],
                          boundary::Array{W,1} = [WT.DEFAULT_BOUNDARY for
                                                  i=1:(m+1)]) where {S <: Integer,
                                                                     T <: Real,
                                                                     W <: WT.WaveletBoundary}
    layeredTransform(m, Xlength, nScales, subsampling, CWTType,
                     averagingLength, averagingType, boundary)
end

# the same for shearlets
function layeredTransform(m::Int, Xsizes::Tuple{<:Integer, <:Integer};
                          nScale::Int=2, nScales::Array{<:Integer,1} = [nScale for
                                                                        i=1:(m+1)],
                          shearLevel::Array{<:Integer, 1} = ceil.(Int,
                                                                  (1:nScale)/2),
                          shearLevels::Array{<:Array{<:Integer, 1}, 1} =
                          [ceil.(Int, (1:nScales[i])/2) for i=1:(m+1)],
                          subsample::S = 2.0, subsamples::Array{<:Real,1} =
                          [Float32(subsample) for i=1:(m+1)], gpu::Bool =
                          false, padding::Bool=true,
                          typeBecomes::DataType=Float32, percentage=.9) where {T<:Number,
                                                                               S<:Number}
    return layeredTransform(m, Xsizes, nScales, subsamples, shearLevels,
                            padding, gpu, percentage, typeBecomes)
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
        @assert arrayType<:AbstractArray{singlentry,N}
        @assert size(data,1)==m+1

        # check output
        singlentry = eltype(output[1])
        arrayType = eltype(output)
        @assert arrayType<:AbstractArray{singlentry,N}
        @assert size(output,1)==m+1
        new(m, k, data, output)
    end
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
    data = [zeros(T, fixDim..., n[i,:]..., prod(q[1:i].-1)) for
            i=1:m+1]
    output = [zeros(T, fixDim...,  n[i,:]..., prod(q[1:(i-1)].-1)) for i=1:m+1]
    return scattered{T,N}(m, k, data, output)
end


# TODO: check performance tips to see if I should add some way of discerning the type of N
function scattered(layers::layeredTransform{S,1}, X::Array{T,N};
                   totalScales=[-1 for i=1:layers.m +1]) where {T <: Real, N,
                                                                S}
    if length(size(X)) == 1
        X = reshape(X, (1,size(X)...));
    end

    n, q, dataSizes, outputSizes, resultingSize = calculateSizes(layers,
                                                                    (-1,-1),
                                                                    size(X),
                                                                    totalScales
                                                                    =
                                                                    totalScales)
    if 1==N
        zerr=[zeros(T, n[i], q[i]) for i=1:layers.m+1]
        output = [zeros(T, n[i+1], q[i]) for i=1:layers.m+1]
    else
        zerr=[zeros(T, size(X)[1:end-1]..., n[i], prod(q[1:i-1].-1)) for
              i=1:layers.m+1]
        output = [zeros(T, size(X)[1:(end-1)]..., n[i+1], prod(q[1:i-1].-1))
                  for i=1:layers.m+1]
    end
    zerr[1][:,:] = X
    return scattered{T,N+1}(layers.m, 1, zerr, output)
end

function scattered(layers::layeredTransform{U,2}, X::Array{T, N};
                   subsample::Bool=true, percentage::Float64=.9) where {U, T <:
                                                                        Number, N,
                                                                        S<:Number} 
    n = sizes(bsplineType(), layers.subsampling, size(X, 1))
    p = sizes(bsplineType(), layers.subsampling, size(X, 2))
    q = [layers.shears[i].nShearlets for i=1:layers.m+1]
    zerr = [zeros(T, n[i], p[i], prod(q[1:i-1].-1)) for i=1:layers.m+1] 
    zerr[1][:,:,1] = X
    reSizingRates = getResizingRates(layers.shears, layers.m,
                                     percentage = percentage)
    if subsample
        output = [zeros(T, reSizingRates[:,i]..., prod(q[1:i-1].-1)) for
                  i=1:layers.m+1] 
    else
        output = [zeros(T, n[i+1], p[i+1], prod(q[1:i-1].-1)) for
                  i=1:layers.m+1]
    end
    return scattered{T, N+1}(layers.m, 2, zerr, output)
end


# Note: if stType is decreasing, this function relies on functions found in Utils.jl
# function scattered(layers::layeredTransform, X::Array{T,N}, stType::String;
#                    totalScales=[-1 for i=1:layers.m+1]) where {T <: Number, N} 
#     @assert stType==fullType() || stType==decreasingType()
#     n, q, dataSizes, outputSizes, resultingSize = calculateSizes(layers,
#                                                                  (-1,-1),
#                                                                  size(X),
#                                                                  totalScales =
#                                                                  totalScales)
#     if stType=="full"
#         zerr=[zeros(T, size(X)[1:end-1]..., n[i], q[i]) for i=1:layers.m+1]
#         output = [zeros(T, size(X)[1:end-1]..., n[i+1], q[i]) for
#                   i=1:layers.m+1] 
#     elseif stType=="decreasing"
#         # brief reminder, smaller index==larger scale
#         counts = [numInLayer(i,layers,q.-1) for i=0:layers.m]
#         zerr=[zeros(T, size(X)[1:end-1]..., n[i], counts[i]) for i=1:layers.m+1]
#         output = [zeros(T, size(X)[1:end-1]..., n[i+1], counts[i]) for i=1:layers.m+1]
#     elseif stType=="collating"
#         zerr=[zeros(T, size(X)[1:end-1]..., n[i], q[1:i-1].-1) for i=1:layers.m+1]
#         output = [zeros(T, size(X)[1:end-1]..., n[i+1], q[1:i-1].-1) for i=1:layers.m+1]
#     end
#     for i in eachindex(view(zerr[1], axes(zerr[1])[1:end-1]..., 1))
#         zerr[1][i,1] = X[i]
#     end
#     scattered{T, N+1}(layers.m, 1, zerr, output)
# end
