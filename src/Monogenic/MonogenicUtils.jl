import FourierFilterFlux:getBatchSize
function getBatchSize(c::MonoConvFFT)
    return c.fftPlan.sz
end

import FourierFilterFlux.cu

function cu(mono::MonoConvFFT{D,OT,F,A,PD,P,T,S,AL}) where {D,OT,F,A,PD,P,T,S,AL}

    # D = mono.D
    # OT = mono.OT
    σ = mono.σ
    boundary = mono.bc

    # trainable = mono.T
    scale = mono.scale

    #averagingLayer = mono.averagingLayer
    averagingLayer = AL

    cuw = cu(mono.weight)
    cuf = cu(mono.fftPlan)


    return MonoConvFFT{D,OT,typeof(σ),typeof(cuw),typeof(boundary),typeof(cuf), T, typeof(scale), typeof(averagingLayer)}(mono.σ, cuw, boundary, cuf, scale, averagingLayer)
end

