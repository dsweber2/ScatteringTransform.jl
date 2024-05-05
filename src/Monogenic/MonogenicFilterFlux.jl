# this file is completely redundant
#include("MonogenicBoundaries.jl");

# MonoConvFFT

# (remark) Similar to ConvFFT
# (remark) but omit bias V, Boundary condition PD , Analyticity An

# (1) D: dimension
#        e.g. N - 1
# (2) OT :: DataType = Float32
#         the output datatype, usually determined by the (space domain) type of the filters.
# (3) F: (not default) Non-linearity 
#         e.g. (σ :: F = typeof(abs))
# (4) A: (not default) weight
#         e.g. (weight :: typeof(w))
# (4) PD: (not default) boundary
#         e.g. (bc :: typeof(boundary))
# (5) P: (not default) Plan
#         Bool=true: use precomputed fft plan(s), as this is a significant cost.
#         Set this to "false" if you have variable batch/channel sizes.
#         e.g. (fftPlan :: typeof(fftPlan))
# (6) T: trainable 
#         Bool=true: The entries are trainable as Flux objects, and so are
#         returned in a "params" call if this is "true".
# (7) S: scale 
# (8) AL: averagingLayer 

#e.g. MonoConvFFT{N - 1,OT,typeof(σ),typeof(w),typeof(fftPlan),trainable, typeof(scale), typeof(averagingLayer)}(σ, w, boundary, fftPlan, scale, averagingLayer)




struct MonoConvFFT{D,OT,F,A,PD,P,T,S,AL}
    σ::F
    weight::A
    bc::PD
    fftPlan::P
    scale::S
    averagingLayer::AL
end

import Base:ndims
ndims(c::MonoConvFFT{D}) where D = D

# MonoConvFFT function
# (remark) Similar to ConvFFT function
# (remark) but omit bias b, Boundary, Analyticity An

function MonoConvFFT(w::AbstractArray{T,N}, originalSize, σ=identity; plan=true, 
        boundary=FourierFilterFlux.Periodic(), dType=Float32, trainable=true, 
        OT=Float32, scale = scale, averagingLayer = false) where {T,N}
    
    @assert length(originalSize) >= N - 1
    if dType <: Complex
        OT = dType
    end

    if length(originalSize) == N - 1
        exSz = (originalSize..., 1) # default number of channels is 1
    else
        exSz = originalSize
    end
    
    netSize, boundary = effectiveSize(exSz[1:N - 1], boundary)
    convDims = (1:(N - 1)...,)

    if dType <: Complex && netSize[1] != size(w, 1)
        wtmp = ifftshift(ifft(w, convDims), convDims)
        wtmp = applyBC(wtmp, boundary, N - 1)
        w = fft(fftshift(wtmp, convDims), convDims)
    elseif dType <: Real && netSize[1] >> 1 + 1 != size(w, 1)
        #wtmp = ifftshift(irfft(w, exSz[1], convDims), convDims)
        #wtmp = ifftshift(ifft(w, convDims), convDims)
        #wtmp, _ = applyBC(wtmp, boundary, N - 1)
        #w = fft(fftshift(wtmp, (convDims)), convDims)
        wtmp = ifft(w, convDims)
        wtmp, _ = applyBC(wtmp, boundary, N - 1)
        w = fft(wtmp, convDims)
    end
    

    if plan
        # change to MonomakePlan
        fftPlan = MonomakePlan(dType, OT, w, exSz, boundary)
    else
        fftPlan = nothing
    end

    return MonoConvFFT{N - 1,OT,typeof(σ),typeof(w),typeof(boundary),typeof(fftPlan),trainable, typeof(scale), typeof(averagingLayer)}(σ, w, boundary, fftPlan, scale, averagingLayer)
end



# MonomakePlan function
# (remark) Similar to makePlan function

function MonomakePlan(dType, OT, w, exSz, boundary)
    N = ndims(w);
    netSize, boundary = effectiveSize(exSz[1:N - 1], boundary);
    convDims = (1:(N-1)...,)
    nullEx = Adapt.adapt(typeof(w), zeros(dType, netSize..., exSz[N:end]...))
    if dType <: Real
        fftPlan = plan_fft(nullEx, convDims)
    else
        null2 = Adapt.adapt(typeof(w), zeros(dType, netSize..., exSz[N:end]...)) .+ 0im
        #fftPlan = (plan_fft(real.(nullEx), convDims), plan_fft(null2, convDims))
        fftPlan = plan_fft(null2, convDims)
    end
end



function Base.show(io::IO, l::MonoConvFFT)
    # stored as a brief description
    if typeof(l.fftPlan) <: Tuple
        sz = l.fftPlan[2]
    else
        sz = l.fftPlan.sz
    end
    es = originalSize(sz[1:ndims(l.weight) - 1], l.bc);
    print(io, "MonoConvFFT[input=$(es), " *
          "nfilters = $(size(l.weight)[end]), " *
          "σ=$(l.σ), " *
          "bc=$(l.bc), " *
          "averagingLayer=$(l.averagingLayer)]")
end


# Define the types of monogenic wavelets
abstract type MonoFilterTypes end

# high-pass Gaussian filter
struct GaussianHP <: MonoFilterTypes end
# low-pass Gaussian filter
struct GaussianLP <: MonoFilterTypes end
include("MonogenicTransform.jl")
include("MonogenicConvFFTConstructors.jl")
export MonogenicLayer
export MonoFilterTypes, GaussianHP, GaussianLP

include("MonogenicUtils.jl");
export getBatchSize

export MonoConvFFT

include("visual_first_layer.jl");
export visual_first_layer

include("visual_second_layer.jl");
export visual_second_layer

include("rotate_image.jl");
export rotate_image

include("inv_rotate_out.jl");
export inv_rotate_out
