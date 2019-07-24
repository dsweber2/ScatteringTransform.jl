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




#################### Reconstruction main functions ##############################


function shearrec2D(coeffs, shearletSystem, P::Future, padded, padBy,
                    resultSize; averaging=false)
    shearrec2D(coeffs, shearletSystem, fetch(P), padded, padBy, resultSize,
               averaging=averaging)
end
    
 
#################### Compute just the Averaging function  ######################

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
function averagingFunction(X::AbstractArray{Complex{T}, N},
                           shearletSystem::Shearlab.Shearletsystem2D{T},
                           P::FFTW.cFFTWPlan{Complex{T},A,B,C}, padded::Bool,
                           padBy::Tuple{Int, Int}) where{T <: Real, A, B, C, N} 
    averagingFunction(view(X,:,:), shearletSystem, P, padded, padBy, true)
end
function averagingFunction(X::AbstractArray{T, N},
                           shearletSystem::Shearlab.Shearletsystem2D{T},
                           P::FFTW.rFFTWPlan{T,A,B,C}, padded::Bool,
                           padBy::Tuple{Int,Int}) where{T <: Real, A, B, C, N}
    averagingFunction(X, shearletSystem, P, padded, padBy, true)
end



function averagingFunction(X, shearletSystem, P::Future, padded, padBy)
    return averagingFunction(X, shearletSystem, fetch(P), padded, padBy)
end


