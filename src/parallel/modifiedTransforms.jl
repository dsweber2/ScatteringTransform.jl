import ContinuousWavelets.cwt
import Wavelets.wavelet
import Wavelets.eltypes
import Wavelets.WT
# so we can write a method that can use either cwt or shearing
function dispatch(dim, toTransform, shear, daughters, fftPlan)
    return cwt(toTransform, shear, daughters, fftPlan)
end

function averaging(dim, toTransform, shear, daughters, fftPlan)
    finalOutput = cwt(toTransform, shear, daughters, fftPlan)
    return dropdims(finalOutput, dims = 2)
end


# a version that can deal with futures, for parallel computing
function cwt(Y::AbstractArray{T,N}, c::CWT{<:Any,<:Any,<:Union{Morlet,Paul}},
    daughters, fftPlan::Future) where {T<:Real,S<:Real,U<:Number,N}
    plrfft, plfft = fetch(fftPlan)
    @debug "size(rfft) = $(size(plrfft)), size(fft) = $(size(plfft))"
    @debug "size of signal = $(size(Y))"
    return cwt(Y, c, daughters, (plrfft, plfft))
end

function cwt(Y::AbstractArray{T,N}, c::CWT{<:Any,<:Any,<:Union{Dog}},
    daughters, fftPlan::Future) where {T<:Real,S<:Real,U<:Number,N}
    pl = fetch(fftPlan)
    return cwt(Y, c, daughters, pl)
end


function numScales(c::CWT, n)
    nOctaves, totalWavelets, sRange, sWidth = ContinuousWavelets.getNWavelets(n, c)
    return totalWavelets
end

function computeWavelets(n1, c; T = Float64, nScales = -1)
    @debug "n1 = $(n1), T = $(T), nScales = $(nScales)"
    @debug "" c
    daughters, ω = ContinuousWavelets.computeWavelets(n1, c, T = T)
    @debug "" nScales
    if nScales > 0
        @debug "computeWavelets: $(nScales) size of daughters is $(size(daughters)), reducing to "
        # "$(1:nScales), i.e. to $(size(daughters[:,1:nScales]))")
        return (SharedArray(daughters[:, 1:nScales]), ω)
    else
        @debug "size of daughters is $(size(daughters))"
        return (SharedArray(daughters), ω)
    end
end
