# TODO make a version of conv that is efficient for small sizes
############################### TRANSFORM TYPES ###############################

# the type T is a type of frame transform that forms the backbone of the transform
struct layeredTransform{T}
    m::Int64 # the number of layers, not counting the zeroth layer
    shears::Array{T} # the array of the transforms; the final of these is used only for averaging, so it has length m+1
    subsampling::Array{Float64,1} # for each layer, the rate of subsampling. There is one of these for layer zero as well, since the output is subsampled, so it should have length m+1
    layeredTransform{T}(m::Int64,shears::Array{T},subsampling::Array{Float64,1}) where T = new(m,shears,subsampling)
end

########################## Shearlet Transforms ##########################
function layeredTransform(m::S, rows::S, cols::S, nScales::Array{S,1}, subsampling::Array{T,1}) where {T<:Real, S<:Integer}
    @assert m+1==size(subsampling,1)
    @assert m+1==size(nScales,1)

    rows = sizes(bilinear,subsampling,(rows)) #TODO: if you add another subsampling method in 2D, this needs to change
    cols = sizes(bilinear,subsampling,(cols)) #TODO: if you add another subsampling method in 2D, this needs to change
    println("rows:$rows cols: $cols nScales: $nScales subsampling: $subsampling")
    shears = [Shearlab.SLgetShearletSystem2D(rows[i], cols[i], nScales[i]) for (i,x) in enumerate(subsampling)]
    layeredTransform{Shearlab.Shearletsystem2D}(m, shears, 1.0*subsampling)
end

# More convenient ways of defining the shattering
layeredTransform(m::S, X::Array{T,2}, nScale::Array{U,1}, subsample::Array{V,1}) where {S<:Integer, T<:Real, U <: Integer, V<:Real} = layeredTransform(m, size(X)..., nScale, subsample)
layeredTransform(m::S, X::Array{T,2}, nScale::Array{S,1}, subsample::S) where {S<:Integer, T<:Real} = layeredTransform(m, size(X)..., nScale, [subsample for i=1:(m+1)])
layeredTransform(m::S, X::Array{T,2}, nScale::S, subsample::Array{S,1}) where {S<:Integer, T<:Real} = layeredTransform(m, size(X)..., [nScale for i=1:(m+1)], subsample)
layeredTransform(m::S, X::Array{T,2}, nScale::S, subsample::S) where {S<:Integer, T<:Real} = layeredTransform(m, size(X)..., [nScale for i=1:(m+1)], [subsample for i=1:(m+1)])

layeredTransform(m::S, X::Array{T,2}, nScale::Array{S,1}) where {S<:Integer, T<:Real} = layeredTransform(m, size(X)..., nScale, [2.0 for i=1:(m+1)])
layeredTransform(m::S, X::Array{T,2}, nScale::S) where {S<:Integer, T <: Real} = layeredTransform(m, size(X)..., [nScale for i=1:(m+1)], [2 for i=1:(m+1)])

layeredTransform(m::S, X::Array{T,2}) where {S<:Integer, T <: Real} = layeredTransform(m, size(X)..., [2 for i=1:(m+1)], [2 for i=1:(m+1)])
layeredTransform(X::Array{T,2}, m::S) where {S<:Integer, T <: Real} = layeredTransform(m, size(X)..., [2 for i=1:(m+1)], [2 for i=1:(m+1)])





################################################################################
######################## Continuous Wavelet Transforms #########################
################################################################################
function layeredTransform(m::S, Xlength::S, nScales::Array{S,1}, subsampling::Array{T,1}, CWTType::WT.ContinuousWaveletClass, averagingLength::Array{S,1}=ceil.(S,nScales*2), averagingType::Array{Symbol,1}=[:Mother for i=1:(m+1)], boundary::Array{W,1}=[WT.DEFAULT_BOUNDARY for i=1:(m+1)]) where {S<:Integer, T<:Real, W<:WT.WaveletBoundary}
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


struct scattered2D{T} <: scattered
  m::Int64 # number of layers, counting the zeroth layer
  data::Array{Array{T,3},1}
  output::Array{Array{T,3},1} # The final averaged results; this is the output from the entire system
  scattered2D{T}(m::Int64, data::Array{Array{T,3},1}, output::Array{Array{T,3},1}) where T<:Number = new{T}(m,data,output)
end
function scattered2D(m::Int64, n::Array{Int64,1}, p::Array{Int64,1}, q::Array{Int64,1},T::DataType)
    @assert (m+1==length(n)) && (m+1==length(p))
    @assert m+1==length(q)
    scattered2D{T}(m+1, [zeros(T, n[i], p[i], prod(q[1:i]-1)) for i=1:m+1], [zeros(Complex128, n[i], p[i], prod(q[1:(i-1)]-1)) for i=1:m])
end
scattered2D(layers::layeredTransform) = scattered2D(layers.m, [layers.shears[i].size[1] for i=1:(layers.m+1)], [layers.shears[i].size[2] for i=1:(layers.m+1)], [size(layers.shears[i].shearlets, 3) for i=1:layers.m+1],Complex128)


function scattered2D(layers::layeredTransform, X::Array{Complex128})
    n = [layers.shears[1].size[1]; [layers.shears[i].size[1] for i=1:(layers.m+1)]]
    p = [layers.shears[1].size[2]; [layers.shears[i].size[2] for i=1:(layers.m+1)]]
    q = [size(layers.shears[i].shearlets,3) for i=1:layers.m+1]
    zerr=[zeros(Complex128, n[i], p[i], prod(q[1:i-1]-1)) for i=1:layers.m+1]
    zerr[1][:,:,1] = X
    output = [zeros(Float64, n[i+1], p[i+1], prod(q[1:i]-1)) for i=1:layers.m+1]
    new(layers.m, zerr, output,Complex128)
end
# TODO: these two should be the same... I believe the complex one is incorrect
function scattered2D(layers::layeredTransform, X::Array{Float64})
    n = [[layers.shears[i].size[1] for i=1:(layers.m+1)]; Int64(ceil.(layers.shears[end].size[1]/layers.subsampling[end]))] # layers.shears[1].size[1];
    p = [[layers.shears[i].size[2] for i=1:(layers.m+1)]; Int64(ceil.(layers.shears[end].size[2]/layers.subsampling[end]))] # layers.shears[1].size[2];
    q = [size(layers.shears[i].shearlets,3) for i=1:layers.m+1]
    zerr=[zeros(Complex128, n[i], p[i], prod(q[1:i-1]-1)) for i=1:layers.m+1]
    zerr[1][:,:,1] = X
    output = [zeros(Float64, n[i+1], p[i+1], prod(q[1:i-1]-1)) for i=1:layers.m+1]
    new(layers.m+1, zerr, output)
end






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

function scattered1D(layers::layeredTransform, X::Array{T,1}, stType::String) where T <: Number
    n = sizes(bspline, layers.subsampling, length(X)) #TODO: if you add another subsampling method in 1D, this needs to change
    q = [numScales(layers.shears[i],n[i]) for i=1:layers.m+1]
    if stType=="full"
        zerr=[zeros(Complex{T}, n[i], prod(q[1:i-1]-1)) for i=1:layers.m+1]
        output = [zeros(Complex{T}, n[i+1], prod(q[1:i-1]-1)) for i=1:layers.m+1]
    elseif stType=="decreasing"
        counts = zeros(Int64,m+1)
        counts[1]=1
        for i=2:length(q)
          for a = 1:q[i-1]
            for b = 1:q[i]
              # check to see if the next layer has a larger scale/smaller frequency
              if b./(layers.shears[i].scalingFactor) <= a./(layers.shears[i-1].scalingFactor)
                counts[i]+=1
              end
            end
          end
        end
        zerr=[zeros(Complex{T}, n[i], counts) for i=1:layers.m+1]
        output = [zeros(Complex{T}, n[i+1], prod(q[1:i-1]-1)) for i=1:layers.m+1]
    elseif stType=="collating"
        zerr=[zeros(Complex{T}, n[i], q[1:i-1]-1) for i=1:layers.m+1]
        output = [zeros(Complex{T}, n[i+1], q[1:i-1]-1) for i=1:layers.m+1]
    end
    zerr[1][:,1] = X
    scattered1D{Complex{T}}(layers.m, zerr, output)
end
