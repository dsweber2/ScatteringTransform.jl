Zygote.@adjoint function ScatteredOut(output,k=1)
    function ∇scattered(δ)
        return (δ.output,nothing)
    end
    ScatteredOut(output,k), ∇scattered
end

Zygote.@adjoint function getindex(F::T, i::Integer) where T <: Scattered
    function getInd_rrule(Ȳ)
        zeroNonRefed = map(ii-> ii-1==i ? Ȳ[1] : zeros(eltype(F.output[ii]),
                                                  size(F.output[ii])...),
                           (1:length(F.output)...,)) 
        ∂F = T(F.m, F.k, zeroNonRefed)
        return ∂F, nothing
    end
    return getindex(F, i), getInd_rrule
end

Zygote.@adjoint function getindex(F::T, inds::AbstractArray) where T<: Scattered
    function getInd_rrule(Ȳ)
        zeroNonRefed = map(ii-> ii-1 in inds ? Ȳ[indexin(ii-1,inds)[1]] :
                           zeros(eltype(F.output[ii]), size(F.output[ii])...),
                           (1:length(F.output)...,)) 
        ∂F = T(F.m, F.k, zeroNonRefed)
        return ∂F, nothing
    end
    return getindex(F, inds), getInd_rrule
end

Zygote.@adjoint function getindex(x::T, p::pathLocs) where T <: Scattered
    function getInd_rrule(Δ)
        zeroNonRefed = map(ii-> zeros(eltype(x.output[ii]),
                                      size(x.output[ii])...),
                           (1:length(x.output)...,))
        ∂x = T(x.m, x.k, zeroNonRefed)
        ∂x[p] = Δ
        return ∂x, nothing
    end
    return getindex(x,p), getInd_rrule
end
