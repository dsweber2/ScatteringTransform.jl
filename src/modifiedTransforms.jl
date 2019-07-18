import Wavelets.cwt
import Wavelets.wavelet
import Wavelets.eltypes
import Wavelets.WT
import Shearlab.Shearletsystem2D

# a version that can deal with futures, for parallel computing
function cwt(Y::AbstractArray{T,N}, c::CFW{W}, daughters::Array{U,2},
             fftPlan::Future; nScales::S=NaN, backOffset::Int=0) where {T<:Real,
                                                                        S<:Real, U<:Number,
                                                                        W<:WT.WaveletBoundary, N}
    pl = fetch(fftPlan)
    return cwt(Y, c, daughters, pl; nScales=nScales, backOffset=backOffset)
end


###################################################################################
#################### Shattering methods ###########################################
###################################################################################

function sheardec2D(X, shearletSystem, P::Future, padded, padBy)
    sheardec2D(X,shearletSystem, fetch(P), padded, padBy)
end

function sheardec2D(X::SubArray{Complex{T}, N},
                    shearletSystem::

                    Shearletsystem2D{T},
                    P::FFTW.cFFTWPlan{Complex{T}, A, B, C},
                    padded::Bool, padBy::Tuple{Int, Int}) where {T <: Real, A,
                                                       B, C, N}
    if shearletSystem.gpu;
        coeffs = AFArray(zeros(Complex{T},
                    size(shearletSystem.shearlets))) 
    else
        coeffs = zeros(Complex{T}, size(shearletSystem.shearlets))
    end

    #compute shearlet coefficients at each scale
    #not that pointwise multiplication in the fourier domain equals convolution
    #in the time-domain
    for j = 1:shearletSystem.nShearlets
        if padded
            # The fourier transform of X
            Xfreq = fftshift(P*(ifftshift(pad(X, padBy))))
            coeffs[:,:,j] = real.(P \ ifftshift(Xfreq.*conj(shearletSystem.shearlets[:,:,j])))[padBy[1] .+ (1:size(X,1)), padBy[2] .+ (1:size(X,2))]
        else
            coeffs[:,:,j] = real.(fftshift(P \ ifftshift(Xfreq.*conj(shearletSystem.shearlets[:,:,j]))))
        end
    end
    return coeffs
end # sheardec2D

function sheardec2D(X::Array{Complex{T}, 2},
           shearletSystem::Shearlab.Shearletsystem2D{T},
           P::FFTW.cFFTWPlan{Complex{T}, A, B, C}, padded::Bool,
           padBy::Tuple{Int, Int}) where {T <: Real, A,
                                          B, C, N}
    sheardec2D(view(X, :, :), shearletSystem, P, padded, padBy)
end


function sheardec2D(X::SubArray{T,N},
                    shearletSystem::Shearlab.Shearletsystem2D{T},
                    P::FFTW.rFFTWPlan{T,B,C,D}, padded::Bool,
                    padBy::Tuple{Int, Int}) where {T<:Real, N, B, C, D}
    if shearletSystem.gpu;
        coeffs = AFArray(zeros(Complex{T},
                               size(shearletSystem.shearlets))) 
    else
        coeffs = zeros(T, size(X)..., size(shearletSystem.shearlets,3))
    end
    
    # compute shearlet coefficients at each scale
    # not that pointwise multiplication in the fourier domain equals convolution
    # in the time-domain
    used1 = padBy[1] .+ (1:size(X)[end-1])
    used2 = padBy[2] .+ (1:size(X)[end])
    for j = 1:shearletSystem.nShearlets
        # The fourier transform of X
        neededShear = conj(shearletSystem.shearlets[:, :, j])
        shearing!(X, neededShear, P,  coeffs, padBy, used1, used2, j)
    end
    return coeffs
end # sheardec2D

function sheardec2D(X::AbstractArray{T,N},
                    shearletSystem::Shearlab.Shearletsystem2D{T}) where {T<:Real, N, B, C, D}
    padBy = getPadBy(shearletSystem)
    fftPlan = createRemoteFFTPlan(size(X)[end-1:end], padBy, T, !(eltype(X)<:Real))
    return sheardec2D(X, shearletSystem, fftPlan, true, padBy)
end



#################### Reconstruction main functions ##############################


function shearrec2D(coeffs::SubArray{T,N},
                    shearletSystem::Shearlab.Shearletsystem2D{T},
                    P::FFTW.rFFTWPlan{T,B,C,D}, padded::Bool,
                    padBy::Tuple{Int, Int}, resultSize::Tuple{Int,Int};
                    averaging::Bool=false) where {T<:Real, N, B, C, D}
    if N==3
        X = zeros(Complex{T}, size(shearletSystem.shearlets)[1:2])
    else
        X = zeros(Complex{T}, size(coeffs)[1:end-3]...,
                  size(shearletSystem.shearlets)[1:2]...)
    end
    
    # sum X in the Fourier domain over the coefficients at each scale/shearing
    used1 = padBy[1] .+ (1:resultSize[1])
    used2 = padBy[2] .+ (1:resultSize[2])
    if averaging
        neededShear = shearletSystem.shearlets[:,:, end]
        unshearing!(X, neededShear, P, coeffs, padBy, size(coeffs,3))
    else
        for j = 1:shearletSystem.nShearlets
            # The fourier transform of X
            neededShear = shearletSystem.shearlets[:, :, j]
            unshearing!(X, neededShear, P,  coeffs, padBy, j)
        end
    end
    realX = zeros(T, size(coeffs)[1:end-1])
    realX!(realX, X, shearletSystem.dualFrameWeights, P, used1, used2)
    return realX
end


function shearrec2D(coeffs, shearletSystem, P::Future, padded, padBy,
                    resultSize; averaging=false)
    shearrec2D(coeffs, shearletSystem, fetch(P), padded, padBy, resultSize,
               averaging=averaging)
end
    
function shearrec2D(coeffs::Array{T,N},
                    shearletSystem::Shearlab.Shearletsystem2D{T},
                    P::FFTW.rFFTWPlan{T,B,C,D}, padded::Bool,
                    padBy::Tuple{Int, Int}, resultSize; averaging::Bool=false) where {T<:Real, N, B, C, D}
    shearrec2D(view(coeffs, axes(coeffs)...), shearletSystem, P, padded, padBy,
               resultSize; averaging = averaging)
end
 




###################### Helper methods shearlets ###################################


function unshearing!(X, neededShear, P, coeffs, padBy, j) where {N,T}
    for i in eachindex(view(X, axes(X)[1:end-2]..., 1, 1))
        cFreq = fftshift( P * (pad(coeffs[i, :, :, j], padBy)))
        X[i, :, :] = X[i, :, :] + cFreq .* neededShear
    end
end

function unshearing!(X::AbstractArray{Complex{T},2}, neededShear, P,
                     coeffs::AbstractArray{T,3}, padBy, j) where {N,T}
    cFreq = fftshift( P * (pad(coeffs[:, :, j], padBy)))
    X[:, :] = X[:, :] + cFreq .* neededShear
end


function realX!(realX, X, duals, P, used1, used2)
    for i in eachindex(view(X, axes(X)[1:end-2]..., 1, 1))
        realX[i, :, :] = real.(fftshift((P \ (ifftshift(X[i,:,:] ./
                                                        (duals))))))[used1,
                                                                     used2]
    end
end
function realX!(realX, X::AbstractArray{T,2}, duals, P, used1, used2) where {T}
    realX[:, :] = real.(fftshift(P \ (ifftshift(X[:,:] ./ (duals)))))[used1,
                                                                      used2]
end


function shearing!(X::AbstractArray{T,N}, neededShear, P, coeffs, padBy, used1,
                   used2, j) where {N,T}
    for i in eachindex(view(X, axes(X)[1:end-2]..., 1, 1))
        Xfreq = fftshift( P * ifftshift(pad(X[i, :, :], padBy)))
        coeffs[i,:,:,j] = real.(P \ ifftshift(Xfreq .* neededShear))[used1,
                                                                         used2]
    end
end

function shearing!(X::AbstractArray{T,2}, neededShear, P, coeffs, padBy, used1,
                   used2, j) where T
    Xfreq =  fftshift( P * ifftshift(pad(X[:, :], padBy)))
    coeffs[:,:,j] = real.(P \ ifftshift(Xfreq .* neededShear))[used1,
                                                               used2]
end


sheardec2D(X::Array{T,2}, shearletSystem::Shearlab.Shearletsystem2D{T},
           P::FFTW.rFFTWPlan{T,B,C,D}, padded::Bool, padBy::Tuple{Int, Int}) where {T<:Real, N, B, C, D} = sheardec2D(view(X,:,:),shearletSystem, P, padded, padBy)


function getPadBy(shearletSystem::Shearlab.Shearletsystem2D{T}) where {T<:Number}
    padBy = (0, 0)
    for sysPad in shearletSystem.support
        padBy = (max(sysPad[1][2]-sysPad[1][1], padBy[1]), max(padBy[2], sysPad[2][2]-sysPad[2][1]))
    end
    return padBy
end


"""
    coeffs = averagingFunction(X,shearletSystem)

compute just the averaging coefficient matrix of the Shearlet transform of the array X. If preFFT is true, then it assumes that you have already taken the fft of X.

general is a dummy variable used to make the more specific versions call the less specific one, since they are used primarily for uniform type checking
...
"""
function averagingFunction(X, shearletSystem, P, padded, padBy, general)
    coeffs = zeros(eltype(X), size(X)..., 1)
    nScale = size(shearletSystem.shearlets,3)
    neededShear = conj(shearletSystem.shearlets[:, :, end])
    used1 = padBy[1] .+ (1:size(X)[end-1])
    used2 = padBy[2] .+ (1:size(X)[end])
    shearing!(X, neededShear, P,  coeffs, padBy, used1, used2, 1)
    return coeffs[:, :]
end # averagingFunction


# make sure the types are done correctly
function averagingFunction(X::SubArray{Complex{T}, N},
                           shearletSystem::Shearlab.Shearletsystem2D{T},
                           P::FFTW.cFFTWPlan{Complex{T},A,B,C}, padded::Bool,
                           padBy::Tuple{Int, Int}) where{T <: Real, A, B, C, N} 
    averagingFunction(view(X,:,:), shearletSystem, P, padded, padBy, true)
end
function averagingFunction(X::SubArray{T, N},
                           shearletSystem::Shearlab.Shearletsystem2D{T},
                           P::FFTW.rFFTWPlan{T,A,B,C}, padded::Bool,
                           padBy::Tuple{Int,Int}) where{T <: Real, A, B, C, N}
    averagingFunction(X, shearletSystem, P, padded, padBy, true)
end



function averagingFunction(X, shearletSystem, P::Future, padded, padBy)
    return averagingFunction(X, shearletSystem, fetch(P), padded, padBy)
end

