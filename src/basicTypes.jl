# TODO make a version of conv that is efficient for small sizes
############################### TRANSFORM TYPES ###############################

# the type T is a type of frame transform that forms the backbone of the transform
struct layeredTransform{T}
    m::Int64 # the number of layers, not counting the zeroth layer
    shears::Array{T} # the array of the transforms; the final of these is used only for averaging, so it has length m+1
    subsampling::Array{Float64,1} # for each layer, the rate of subsampling. There is one of these for layer zero as well, since the output is subsampled, so it should have length m+1
    layeredTransform{T}(m::Int64,shears::Array{T},subsampling::Array{Float64,1}) where T = new(m,shears,subsampling)
end




################################################################################
######################## Continuous Wavelet Transforms #########################
################################################################################
function layeredTransform(m::S, Xlength::S, nScales::Array{T,1}, subsampling::Array{T,1}, CWTType::WT.ContinuousWaveletClass, averagingLength::Array{S,1}=ceil.(S,nScales*2), averagingType::Array{Symbol,1}=[:Mother for i=1:(m+1)], boundary::Array{W,1}=[WT.DEFAULT_BOUNDARY for i=1:(m+1)]) where {S<:Integer, T<:Real, W<:WT.WaveletBoundary}
    @assert m+1==size(subsampling,1)
    @assert m+1==size(nScales,1)

    Xlength = sizes(bspline,subsampling,(Xlength)) #TODO: if you add another subsampling method in 1D, this needs to change
    println("Vector Lengths: $Xlength nScales: $nScales subsampling: $subsampling")
    shears = [wavelet(CWTType, nScales[i], averagingLength[i], averagingType[i], boundary[i]) for (i,x) in enumerate(subsampling)]
    layeredTransform{typeof(shears[1])}(m, shears, 1.0*subsampling)
end

# versions with explicit name calling
function layeredTransform(m::S, Xlength::S; nScales::Array{S,1}=[8 for i=1:(m+1)], subsampling::Array{T,1}=[2 for i=1:(m+1)], CWTType::WT.ContinuousWaveletClass=WT.morl, averagingLength::Array{S,1}=ceil.(S, nScales/2), averagingType::Array{Symbol,1}=[:Mother for i=1:(m+1)], boundary::Array{W,1}=[WT.DEFAULT_BOUNDARY for i=1:(m+1)]) where {S<:Integer, T<:Real, W<:WT.WaveletBoundary}
    layeredTransform(m, Xlength, nScales, subsampling, CWTType, averagingLength, averagingType, boundary)
end
function layeredTransform(m::S, X::Vector{U}; nScales::Array{S,1}=[8 for i=1:(m+1)], subsampling::Array{T,1}=[2 for i=1:(m+1)], CWTType::WT.ContinuousWaveletClass=WT.morl, averagingLength::Array{S,1}=ceil.(S, nScales/2), averagingType::Array{Symbol,1}=[:Mother for i=1:(m+1)], boundary::Array{W,1}=[WT.DEFAULT_BOUNDARY for i=1:(m+1)]) where {S<:Integer, T<:Real, U<:Real, W<:WT.WaveletBoundary}
    layeredTransform(m, length(X), nScales, subsampling, CWTType, averagingLength, averagingType, boundary)
end

# the methods where we are given a transform
layeredTransform(m::S, Xlength::S, nScales::S, subsampling::Array{T,1}, CWTType::WT.ContinuousWaveletClass) where {S<:Integer, T<:Real} = layeredTransform(m, Xlength, [nScales for i=1:(m+1)], subsampling, CWTType)
layeredTransform(m::S, Xlength::S, nScales::Array{S,1}, subsampling::T, CWTType::WT.ContinuousWaveletClass) where {S<:Integer, T<:Real} = layeredTransform(m, Xlength, nScales, [subsampling for i=1:m+1], CWTType)
layeredTransform(m::S, Xlength::S, nScales::S, subsampling::T, CWTType::WT.ContinuousWaveletClass) where {S<:Integer, T<:Real} = layeredTransform(m, Xlength, [nScales for i=1:(m+1)], [subsampling for i=1:(m+1)], CWTType)


layeredTransform(m::S, X::Vector{U}, nScales::Array{S,1}, subsampling::Array{T,1}, CWTType::WT.ContinuousWaveletClass) where {S<:Integer,T<:Real,U<:Real} = layeredTransform(m, length(X), nScales, subsampling, CWTType)
layeredTransform(m::S, X::Vector{U}, nScales::S, subsampling::Array{T,1}, CWTType::WT.ContinuousWaveletClass) where {S<:Integer, T<:Real,U<:Real} = layeredTransform(m, length(X), [nScales for i=1:(m+1)], subsampling, CWTType)
layeredTransform(m::S, X::Vector{U}, nScales::Array{S,1}, subsampling::T, CWTType::WT.ContinuousWaveletClass) where {S<:Integer, T<:Real,U<:Real} = layeredTransform(m, length(X), nScales, [subsampling for i=1:(m+1)], CWTType)
layeredTransform(m::S, X::Vector{U}, nScales::S, subsampling::T, CWTType::WT.ContinuousWaveletClass) where {S<:Integer, T<:Real,U<:Real} = layeredTransform(m, length(X), [nScales for i=1:(m+1)], [subsampling for i=1:(m+1)], CWTType)


# default subsampling
layeredTransform(m::S, X::Vector{U}, nScales::Array{T,1}, CWTType::WT.ContinuousWaveletClass) where {S<:Integer, T<:Real,U<:Real} = layeredTransform(m, length(X), nScales, [2 for i=1:(m+1)], CWTType)
layeredTransform(m::S, X::Vector{T}, nScales::S, CWTType::WT.ContinuousWaveletClass) where {S<:Integer, T<:Real} = layeredTransform(m, length(X), [nScales for i=1:(m+1)], [2 for i=1:(m+1)], CWTType)
layeredTransform(m::S, lengthX::T, nScales::Array{U,1}, CWTType::WT.ContinuousWaveletClass) where {S<:Integer, T<:Real, U<:Real} = layeredTransform(m, lengthX, nScales, [2 for i=1:(m+1)], CWTType)
layeredTransform(m::S, lengthX::T, nScales::U, CWTType::WT.ContinuousWaveletClass) where {S<:Integer, T<:Real, U<:Real} = layeredTransform(m, lengthX, [nScales for i=1:(m+1)], [2 for i=1:(m+1)], CWTType)


# default subsampling and scales
layeredTransform(m::S, X::Vector{T}, CWTType::WT.ContinuousWaveletClass) where {S<:Integer, T<:Real} = layeredTransform(m, length(X), [8 for i=1:(m+1)], [2 for i=1:(m+1)], CWTType)





####################################
# Without a transform, 1D data defaults to a morlet wavelet
layeredTransform(m::S, Xlength::T, nScales::Array{U,1}, subsampling::Array{U,1}) where {S<:Integer, T<:Real, U<:Real} = layeredTransform(m, Xlength, nScales, subsampling, WT.morl)
layeredTransform(m::S, Xlength::S, nScales::S, subsampling::Array{T,1}) where {S<:Integer, T<:Real} = layeredTransform(m, Xlength, [nScales for i=1:(m+1)], subsampling, WT.morl)
layeredTransform(m::S, Xlength::S, nScales::Array{S,1}, subsampling::T) where {S<:Integer, T<:Real} = layeredTransform(m, Xlength, nScales, [subsampling for i=1:(m+1)], WT.morl)
layeredTransform(m::S, Xlength::S, nScales::S, subsampling::T) where {S<:Integer, T<:Real} = layeredTransform(m, Xlength, [nScales for i=1:(m+1)], [subsampling for i=1:(m+1)], WT.morl)

layeredTransform(m::S, X::Vector{T}, nScales::Array{U,1}, subsampling::Array{U,1}) where {S<:Integer, T<:Real, U<:Real} = layeredTransform(m, length(X), nScales, subsampling, WT.morl)
layeredTransform(m::S, X::Vector{U}, nScales::S, subsampling::Array{T,1}) where {S<:Integer, T<:Real, U<:Real} = layeredTransform(m, length(X), [nScales for i=1:(m+1)], subsampling, WT.morl)
layeredTransform(m::S, X::Vector{U}, nScales::Array{S,1}, subsampling::T) where {S<:Integer, T<:Real, U<:Real} = layeredTransform(m, length(X), nScales, [subsampling for i=1:(m+1)], WT.morl)
layeredTransform(m::S, X::Vector{U}, nScales::S, subsampling::T) where {S<:Integer, T<:Real, U<:Real} = layeredTransform(m, length(X), [nScales for i=1:(m+1)], [subsampling for i=1:(m+1)], WT.morl)

layeredTransform(m::S, Xlength::S, nScales::T) where {S<:Integer, T<:Real} = layeredTransform(m, Xlength, [nScales for i=1:(m+1)], [2 for i=1:m+1], WT.morl)
layeredTransform(m::S, Xlength::T, nScales::Array{U,1}) where {S<:Integer, T<:Real, U<:Real} = layeredTransform(m, Xlength, nScales, [2 for i=1:(m+1)], WT.morl)

layeredTransform(m::S, X::Vector{T}, nScales::Array{U,1}) where {S<:Integer, T<:Real, U<:Real} = layeredTransform(m, length(X), nScales, [2 for i=1:(m+1)], WT.morl)
layeredTransform(m::S, X::Vector{U}, nScales::T) where {S<:Integer, T<:Real, U<:Real} = layeredTransform(m, length(X), [nScales for i=1:(m+1)], [2 for i=1:(m+1)], WT.morl)

layeredTransform(m::S, Xlength::S) where {S<:Integer} = layeredTransform(m, Xlength, [8 for i=1:(m+1)], [2 for i=1:m+1], WT.morl)
layeredTransform(m::S, X::Vector{U}) where {S<:Integer,U<:Real} = layeredTransform(m, length(X), [8 for i=1:(m+1)], [2 for i=1:(m+1)], WT.morl)




# TODO: need to test that the sizes come out right
# TODO: tests for the data collators and the subsampling
# TODO: Write a version of this that accomodates things that are too big to hold in memory
# TODO: someday, make this more efficient (precompute the filters, for example)

abstract type scattered end


struct scattered1D{T} <: scattered
  m::Int64 # number of layers, counting the zeroth layer
  data::Array{Array{T,2},1} #first dimension is time, second is path/frequency
  output::Array{Array{T,2},1} # The final averaged results; this is the output from the entire system
    scattered1D{T}(m::Int64, data::Array{Array{T,2},1}, output::Array{Array{T,2},1}) where T = new(m,data,output)
end


function scattered1D(m::Int64, n::Array{Int64,1}, q::Array{Int64,1},T::DataType)
    @assert m+1==length(n)
    @assert m+1==length(q)
    scattered1D{T}(m, [zeros(T, n[i], prod(q[1:i]-1)) for i=1:m+1], [zeros(T, n[i], p[i], prod(q[1:(i-1)]-1)) for i=1:m])
end
function scattered1D(layers::layeredTransform, X::Array{T,1}) where T <: Number
    n = sizes(bspline, layers.subsampling, length(X)) #TODO: if you add another subsampling method in 1D, this needs to change
    q = [numScales(layers.shears[i],n[i]) for i=1:layers.m+1]
    zerr=[zeros(Complex{T}, n[i], prod(q[1:i-1]-1)) for i=1:layers.m+1]
    zerr[1][:,1] = X
    output = [zeros(Complex{T}, n[i+1], prod(q[1:i-1]-1)) for i=1:layers.m+1]
    scattered1D{Complex{T}}(layers.m, zerr, output)
end

# Note: if stType is decreasing, this function relies on functions found in Utils.jl
function scattered1D(layers::layeredTransform, X::Array{T,1}, stType::String) where T <: Number
    n = sizes(bspline, layers.subsampling, length(X)) #TODO: if you add another subsampling method in 1D, this needs to change
    q = [numScales(layers.shears[i], n[i]) for i=1:layers.m+1]
    if stType=="full"
        zerr=[zeros(Complex{T}, n[i], prod(q[1:i-1]-1)) for i=1:layers.m+1]
        output = [zeros(Complex{T}, n[i+1], prod(q[1:i-1]-1)) for i=1:layers.m+1]
    elseif stType=="decreasing"
        # brief reminder, smaller index==larger scale
        counts = [numInLayer(i,layers,q-1) for i=0:layers.m]
        zerr=[zeros(Complex{T}, n[i], counts[i]) for i=1:layers.m+1]
        output = [zeros(Complex{T}, n[i+1], counts[i]) for i=1:layers.m+1]
    elseif stType=="collating"
        zerr=[zeros(Complex{T}, n[i], q[1:i-1]-1) for i=1:layers.m+1]
        output = [zeros(Complex{T}, n[i+1], q[1:i-1]-1) for i=1:layers.m+1]
    end
    zerr[1][:,1] = X
    scattered1D{Complex{T}}(layers.m, zerr, output)
end
