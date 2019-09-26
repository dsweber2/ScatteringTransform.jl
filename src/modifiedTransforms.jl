import Wavelets.cwt
import Wavelets.wavelet
import Wavelets.eltypes
import Wavelets.WT
import Shearlab.Shearletsystem2D
# so we can write a method that can use either cwt or shearing
function dispatch(dim, toTransform, shear, daughters, fftPlan)
    if dim==1
        return cwt(toTransform, shear, daughters, fftPlan)
    elseif dim==2
        return output = sheardec2D(toTransform, shear, fftPlan)
    end
end

function averaging(dim, toTransform, shear, daughters, fftPlan)
    if dim==1
        finalOutput = cwt(toTransform, shear, daughters, fftPlan)
        return dropdims(finalOutput, dims=2)
    elseif dim==2
        return averagingFunction(toTransform, shear, fftPlan)
    end
end


# a version that can deal with futures, for parallel computing
function cwt(Y::AbstractArray{T,N}, c::CFW{<:Any,<:Any,<:Union{WT.Morlet,
                                                               WT.Paul}},
             daughters, fftPlan::Future) where {T<:Real, S<:Real, U<:Number,
                                                N}
    plrfft, plfft = fetch(fftPlan)
    # println("size(rfft) = $(size(plrfft)), size(fft) = $(size(plfft))")
    # println("size of signal = $(size(Y))")
    return cwt(Y, c, daughters, plrfft, plfft)
end

function cwt(Y::AbstractArray{T,N}, c::CFW{<:Any,<:Any,<:Union{WT.Dog}},
             daughters, fftPlan::Future) where {T<:Real, S<:Real, U<:Number, N}
    pl = fetch(fftPlan)
    return cwt(Y, c, daughters, pl)
end


function numScales(c::CFW, n)
    n= Int(n)
    isAve = (c.averagingLength > 0 && !(typeof(c.averagingType) <: WT.NoAve)) ? 1 : 0
    nOctaves = log2(max(n, 2)) - c.averagingLength
    nWaveletsInOctave = reverse([max(1, round(Int, c.scalingFactor /
                                              x^(c.decreasing))) for
                                 x=1:round(Int, nOctaves)])
    return round(Int, sum(nWaveletsInOctave) + isAve)
end

function numScales(c::CFW, n, i)
    nnn = numScales(c,n)
    return nnn[i]
end

function numScales(shearletSystem::Shearlab.Shearletsystem2D)
    return size(shearletSystem.shearlets)[end]-1
end

function numScales(shearletSystem::Shearlab.Shearletsystem2D, n)
    return size(shearletSystem[i].shearlets)[end]-1
end

function numScales(shearletSystem::Shearlab.Shearletsystem2D, n, i)
    return size(shearletSystem[i].shearlets)[end]-1
end

function computeWavelets(n1, c; T=Float64, nScales=-1)
    #println("n1 = $(n1), T = $(T), nScales = $(nScales)")
    #println("$(c)")
    daughters,ω = Wavelets.computeWavelets(n1,c,T=T)
    #println("nScales = $(nScales)")
    if nScales>0
        #println("computeWavelets: $(nScales) size of daughters is $(size(daughters)), reducing to "*
        #"$(1:nScales), i.e. to $(size(daughters[:,1:nScales]))")
        return (SharedArray(daughters[:,1:nScales]), ω)
    else
        #println("size of daughters is $(size(daughters))")
        return (SharedArray(daughters), ω)
    end
end
###################################################################################
#################### Shattering methods ###########################################
###################################################################################

function sheardec2D(X, shearletSystem, P::Future)
    Shearlab.sheardec2D(X, shearletSystem, P = fetch(P))
end




#################### Reconstruction main functions ##############################


function shearrec2D(coeffs, shearletSystem, P::Future,
                    resultSize; averaging=false)
    Shearlab.shearrec2D(coeffs, shearletSystem, P=fetch(P))
end
    
 
#################### Compute just the Averaging function  ######################

"""
    coeffs = averagingFunction(X,shearletSystem)

compute just the averaging coefficient matrix of the Shearlet transform of the array X. If preFFT is true, then it assumes that you have already taken the fft of X.

general is a dummy variable used to make the more specific versions call the less specific one, since they are used primarily for uniform type checking
...
"""
function averagingFunction(X, shearletSystem, P, general)
    padBy = shearletSystem.padBy
    coeffs = zeros(eltype(X), size(X)..., 1)
    nScale = size(shearletSystem.shearlets,3)
    neededShear = conj(shearletSystem.shearlets[:, :, end])
    used1 = padBy[1] .+ (1:size(X, 1))
    used2 = padBy[2] .+ (1:size(X, 2))
    Shearlab.shearing!(X, neededShear, P,  coeffs, padBy, used1, used2, 1) # the one is just for accessing coeffs
    return coeffs[:, :]
end # averagingFunction


# make sure the types are done correctly
function averagingFunction(X::AbstractArray{Complex{T}, N},
                           shearletSystem::Shearlab.Shearletsystem2D{T},
                           P::FFTW.cFFTWPlan{Complex{T},A,B,C}) where {T <:
                                                                       Real, A,
                                                                       B, C, N}  
    averagingFunction(X, shearletSystem, P, true)
end
function averagingFunction(X::AbstractArray{T, N},
                           shearletSystem::Shearlab.Shearletsystem2D{T},
                           P::FFTW.rFFTWPlan{T,A,B,C}) where {T <: Real, A, B,
                                                             C, N} 
    averagingFunction(X, shearletSystem, P, true)
end



function averagingFunction(X, shearletSystem, P::Future)
    return averagingFunction(X, shearletSystem, fetch(P))
end


