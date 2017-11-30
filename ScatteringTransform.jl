# TODO make a version of conv that is efficient for small sizes
using Wavelets, Plots, DSP, Shearlab


############################### TRANSFORM TYPES ###############################

# the type T is a type of frame transform that forms the backbone of the transform
struct layeredTransform{T}
    m::Int64 # the number of layers, not counting the zeroth layer
    shears::Array{T} # the array of the transforms
    subsampling::Array{Float64,1} # for each layer, the rate of subsampling.
end

########################## Shearlet Transforms ##########################
function layeredTransform(m::S, rows::S, cols::S, nScales::Array{S,1}, subsampling::Array{T,1}) where {T<:Real, S<:Integer}
    @assert m+1==size(subsampling,1)
    @assert m+1==size(nScales,1)

    rows = [Int64( foldl((x,y)->ceil(x/y), rows, subsampling[1:i])) for i=0:length(subsampling)-1]
    cols = [Int64(foldl((x,y)->ceil(x/y), cols, subsampling[1:i])) for i=0:length(subsampling)-1]
    println("rows:$rows cols: $cols nScales: $nScales subsampling: $subsampling")
    shears = [Shearlab.SLgetShearletSystem2D(rows[i], cols[i], nScales[i]) for (i,x) in enumerate(subsampling)]
    layeredTransform{Shearlab.Shearletsystem2D}(m, shears, subsampling)
end

# More convenient ways of defining the shattering
layeredTransform(m::S, rows::S, cols::S, nScales::S, subsampling::Array{T,1}) where {T<:Real, S<:Integer} = layeredTransform(m, rows, cols, [nScales for i=1:(m+1)], subsampling)
layeredTransform(m::S, rows::S, cols::S, nScales::Array{S,1}, subsample::T) where {T<:Real, S<:Integer} = layeredTransform(m, rows, cols, nScales, [subsample for i=1:m+1])
layeredTransform(m::S, rows::S, cols::S, nScales::S, subsample::T) where {T<:Real, S<:Integer} = layeredTransform(m, rows, cols, [nScales for i=1:(m+1)], [subsample for i=1:(m+1)])

layeredTransform(m::T, rows::T, cols::T, nScales::Array{T,1}) where T<:Integer = layeredTransform(m, rows, cols, nScales, [2.0 for i=1:(m+1)])
layeredTransform(m::T, rows::T, cols::T, nScales::T) where T<:Integer = layeredTransform(m, rows, cols, [nScales for i=1:(m+1)], [2.0 for i=1:(m+1)])

layeredTransform(m::Int64, X::Array{T,2}, nScale::Array{Int64,1}) where T<:Real = layeredTransform(m, size(X)[1], size(X)[2], nScale)
layeredTransform(m::Int64, X::Array{T,2}, nScale::Int64) where T<:Real = layeredTransform(m, size(X)[1], size(X)[2], [nScale for i=1:(m+1)])

layeredTransform(m::Int64, X::Array{T,2}, nScale::Array{Int64,1}, subsample::S) where {T, S <: Real} = layeredTransform(m, size(X)[1], size(X)[2], nScale, subsample)
layeredTransform(m::Int64, X::Array{T,2}, nScale::Int64, subsample::S) where {T, S <: Real} = layeredTransform(m, size(X)[1], size(X)[2], [nScale for i=1:(m+1)], subsample)

layeredTransform(m::Int64, X::Array{T,2}) where {T<:Real} = layeredTransform(m, size(X)[1], size(X)[2], 2)

layeredTransform(X::Array{T,2},m::Int64) where {T<:Real} = layeredTransform(m, size(X)[1], size(X)[2], 2)

######################## Continuous Wavelet Transforms #########################
function layeredTransform(m::S, Xlength::S, nScales::Array{S,1}, subsampling::Array{T,1},CWTType::WT.ContinuousWaveletClass) where {S<:Integer, T<:Real}
    @assert m+1==size(subsampling,1)
    @assert m+1==size(nScales,1)

    Xlength = [Int64( foldl((x,y)->ceil(x/y), Xlength, subsampling[1:i])) for i=0:length(subsampling)-1]
    println("Vector Lengths: $Xlength nScales: $nScales subsampling: $subsampling")
    shears = [wavelet(CWTType,nScales[i]) for (i,x) in enumerate(subsampling)]
    layeredTransform{typeof(shears[1])}(m, shears, subsampling)
end

# the methods where we are given a transform
layeredTransform(m::S, Xlength::S, nScales::S, subsampling::Array{T,1}, CWTType::WT.ContinuousWaveletClass) where {S<:Integer, T<:Real} = layeredTransform(m, Xlength, [nScales for i=1:(m+1)], subsampling, CWTType)
layeredTransform(m::S, Xlength::S, nScales::Array{S,1}, subsampling::T, CWTType::WT.ContinuousWaveletClass) where {S<:Integer, T<:Real} = layeredTransform(m, Xlength, nScales, [subsampling for i=1:(m+1)], CWTType)
layeredTransform(m::S, Xlength::S, nScales::S, subsampling::T, CWTType::WT.ContinuousWaveletClass) where {S<:Integer, T<:Real} = layeredTransform(m, Xlength, [nScales for i=1:(m+1)], [subsampling for i=1:(m+1)], CWTType)
layeredTransform(m::S, Xlength::S, nScales::Array{S,1}, subsampling::Array{U,1}, CWTType::WT.ContinuousWaveletClass) where {S<:Integer, T,U<:Real} = layeredTransform(m, length(X), nScales, subsampling, CWTType)

layeredTransform(m::S, X::Vector{U}, nScales::Array{S,1}, subsampling::Array{U,1}, CWTType::WT.ContinuousWaveletClass) where {S<:Integer, T,U<:Real} = layeredTransform(m, length(X), nScales, subsampling, CWTType)
layeredTransform(m::S, X::Vector{U}, nScales::S, subsampling::Array{T,1}, CWTType::WT.ContinuousWaveletClass) where {S<:Integer, T,U<:Real} = layeredTransform(m, length(X), [nScales for i=1:(m+1)], subsampling, CWTType)
layeredTransform(m::S, X::Vector{U}, nScales::Array{S,1}, subsampling::T, CWTType::WT.ContinuousWaveletClass) where {S<:Integer, T,U<:Real} = layeredTransform(m, length(X), nScales, [subsampling for i=1:(m+1)], CWTType)
layeredTransform(m::S, X::Vector{U}, nScales::S, subsampling::T, CWTType::WT.ContinuousWaveletClass) where {S<:Integer, T,U<:Real} = layeredTransform(m, length(X), [nScales for i=1:(m+1)], [subsampling for i=1:(m+1)], CWTType)

# default subsampling
layeredTransform(m::S, X::Vector{U}, nScales::Array{S,1}, CWTType::WT.ContinuousWaveletClass) where {S<:Integer, T,U<:Real} = layeredTransform(m, length(X), nScales, [2 for i=1:(m+1)], CWTType)
layeredTransform(m::S, X::Vector{U}, nScales::S, CWTType::WT.ContinuousWaveletClass) where {S<:Integer, T,U<:Real} = layeredTransform(m, length(X), [nScales for i=1:(m+1)], [2 for i=1:(m+1)], CWTType)
layeredTransform(m::S, lengthX::S, nScales::Array{S,1}, CWTType::WT.ContinuousWaveletClass) where {S<:Integer, T,U<:Real} = layeredTransform(m, lengthX, nScales, [2 for i=1:(m+1)], CWTType)
layeredTransform(m::S, lengthX::S, nScales::S, CWTType::WT.ContinuousWaveletClass) where {S<:Integer, T,U<:Real} = layeredTransform(m, lengthX, [nScales for i=1:(m+1)], [2 for i=1:(m+1)], CWTType)

# default subsampling and scales
layeredTransform(m::S, X::Vector{U}, CWTType::WT.ContinuousWaveletClass) where {S<:Integer, T,U<:Real} = layeredTransform(m, length(X), [8 for i=1:(m+1)], [2 for i=1:(m+1)], CWTType)

####################################
# Without a transform, 1D data defaults to a morlet wavelet
layeredTransform(m::S, Xlength::S, nScales::S, subsampling::Array{T,1}) where {S<:Integer, T<:Real} = layeredTransform(m, Xlength, [nScales for i=1:(m+1)], subsampling, WT.morl)
layeredTransform(m::S, Xlength::S, nScales::Array{S,1}, subsampling::T) where {S<:Integer, T<:Real} = layeredTransform(m, Xlength, nScales, [subsampling for i=1:(m+1)], WT.morl)
layeredTransform(m::S, Xlength::S, nScales::S, subsampling::T) where {S<:Integer, T<:Real} = layeredTransform(m, Xlength, [nScales for i=1:(m+1)], [subsampling for i=1:(m+1)], WT.morl)
layeredTransform(m::S, Xlength::S, nScales::Array{S,1}, subsampling::Array{U,1}) where {S<:Integer, T,U<:Real} = layeredTransform(m, length(X), nScales, subsampling, WT.morl)

layeredTransform(m::S, X::Vector{U}, nScales::Array{S,1}, subsampling::Array{U,1}) where {S<:Integer, T,U<:Real} = layeredTransform(m, length(X), nScales, subsampling, WT.morl)
layeredTransform(m::S, X::Vector{U}, nScales::S, subsampling::Array{T,1}) where {S<:Integer, T,U<:Real} = layeredTransform(m, length(X), [nScales for i=1:(m+1)], subsampling, WT.morl)
layeredTransform(m::S, X::Vector{U}, nScales::Array{S,1}, subsampling::T) where {S<:Integer, T,U<:Real} = layeredTransform(m, length(X), nScales, [subsampling for i=1:(m+1)], WT.morl)
layeredTransform(m::S, X::Vector{U}, nScales::S, subsampling::T) where {S<:Integer, T,U<:Real} = layeredTransform(m, length(X), [nScales for i=1:(m+1)], [subsampling for i=1:(m+1)], WT.morl)

layeredTransform(m::S, Xlength::S, nScales::S) where {S<:Integer, T<:Real} = layeredTransform(m, Xlength, [nScales for i=1:(m+1)], [2 for i=1:m+1], WT.morl)
layeredTransform(m::S, Xlength::S, nScales::Array{S,1}) where {S<:Integer, T,U<:Real} = layeredTransform(m, length(X), nScales, [2 for i=1:(m+1)], WT.morl)

layeredTransform(m::S, X::Vector{U}, nScales::Array{S,1}) where {S<:Integer, T,U<:Real} = layeredTransform(m, length(X), nScales, [2 for i=1:(m+1)], WT.morl)
layeredTransform(m::S, X::Vector{U}, nScales::S) where {S<:Integer, T,U<:Real} = layeredTransform(m, length(X), [nScales for i=1:(m+1)], [2 for i=1:(m+1)], WT.morl)

layeredTransform(m::S, Xlength::S) where {S<:Integer, T<:Real} = layeredTransform(m, Xlength, [8 for i=1:(m+1)], [2 for i=1:m+1], WT.morl)
layeredTransform(m::S, X::Vector{U}) where {S<:Integer, T,U<:Real} = layeredTransform(m, length(X), [8 for i=1:(m+1)], [2 for i=1:(m+1)], WT.morl)

# tests for the various forms of layeredTransform for the 1D ContinuousWaveletClass transforms
f = randn(1000)
layeredTransform(3, length(f), [2, 2, 2, 2], [2, 2, 2, 2], WT.morl)
layeredTransform(3, f, [8, 4, 2, 1], [2, 2, 2, 2])
layeredTransform(3, f, [8, 4, 2, 1], 2)
layeredTransform(3, length(f), 2, 2)

layeredTransform(f,2)
WT.CFW
# tests for the various forms of layeredTransform for the ShatteringTransform
layeredTransform(3,1000,1000,3,[1.0, 1.0, 1.0, 1.0])
layeredTransform(2,1000,1000,3,2)
layeredTransform(2,1000,1000,3)
X = randn(1000,1000)
layeredTransform(2,X,2)
layeredTransform(2,X,2,3)
layeredTransform(2,X)
layeredTransform(X,2)



# TODO: need to test that the sizes come out right
# TODO: add a version where the number of scales changes per layer
# TODO: someday, make this more efficient (precompute the filters, for example)

ShatteringTransform()
# mucking around with Wavelets.jl
n=2053
plot(x)
awef = cwt(x, wavelet(WT.morl))
heatmap(abs.(awef))
x = testfunction(n,"Doppler")
wt = wavelet(WT.db2)
y = dwt(x, wt, maxtransformlevels(x))
y2 = dwt(x, WT.scale(wt,1/2))
y4 = dwt(x, WT.scale(wt,1/4))
plot(y, label="wt")
plot(y2, label="wt2")
plot(y4, label="wt4")
d,l = wplotdots(y, 0.1, n)
Ap = wplotim(y)
heatmap(A)
heatmap(Ap)
plot(y,label="discrete")
plot(x)


N = 2^10


h = WT.qmf(wavelet(WT.db2))
plot(h)

# Find the initial points
H = Array{eltype(h)}(length(h),length(h))
for i=1:length(h)
    for j=1:length(h)
        if 2(i-1)-(j-1) >= 0 && 2i-j <= length(h)
            H[i,j] = h[2i-j]
        else
            H[i,j] = 0
        end
    end
end
eigval, evec = eig(H)
ϕ = evec[:,eigval.==1/sqrt(2)]
ϕ = ϕ./sum(ϕ)
plot(ϕ)
plot(0:3,h)
plot!(0:4/64:3, resample(resample(resample(resample(h,2.025),2.025),2.025),2.025))
FIRFilter(h,2//1)
resample(h, 2//1, resample_filter(2//1))
FIRRational(h,2//1)
resample(resample(h,2//1),2//1)
plot(resample(resample(h,2.0),2.0))
plot(filt(FIRFilter(2//1), h))
scatter!(1:2:8,h)
# keep going until we get to over N points
ϕp = zeros(eltype(h), 2*length(ϕ))
# upsample h
hp = upsample(h)
# smooth out the zeros we padded h with
hp =
plot(sinc(-π:.05:π))
plot(hp⋆ϕ/sum(hp⋆ϕ))
h
for m=1:length(ϕp)
    # these are from the previous layer
    if m%2==1
        ϕp[m] = ϕ[div(m+1,2)]
    else
        for n=1:length(ϕ)
            if 2(n-1)-(m-1) >= 0 && 2n-m <= length(h)
                ϕp[m] += h[2n-m].*ϕ[div(n,2)]
            end
        end
    end
end
ϕp = sum(ϕp)*ϕp
plot(ϕp)
ϕp
# resample back down to N points
eval==sqrt(2)
plot(evec[:,2])
ϕ =
1/sqrt(sqrt(2))
svd(H)




sqrt(2)*h
