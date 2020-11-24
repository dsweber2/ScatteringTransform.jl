# function rrule(::typeof(RationPool),resSize,k=2; nExtraDims=2, poolType = MeanPool)
#     function RationalPoolΔ(δ)
#         println("problems")
#         println(NO_FIELDS)
#         return NO_FIELDS
#     end
    
#     return RationPool(resSize,k; nExtraDims=2, poolType = MeanPool), RationalPoolΔ
# end


# function rrule(::typeof(ScatteredOut), m, k, output)
#     function ∇scattered(δ)
#         println("base constructor ",typeof(δ))
#         return (δ.output, )
#     end
#     ScatteredOut(m,k,output), ∇scattered
# end
#@non_differentiable ScatteredOut(::Any)
Zygote.@adjoint function ScatteredOut(output,k=1)
    function ∇scattered(δ)
        return (δ.output,)
    end
    ScatteredOut(output,k), ∇scattered
end

# function rrule(::typeof(ScatteredOut), output,k=1)
#     function ∇scattered(δ)
#         println("abridged constructor ",typeof(δ))
#         return (δ.output, )
#     end
#     ScatteredOut(output,k), ∇scattered
# end

# function rrule(::typeof(ScatteredOut), m,k,fixDim::Array{<:Real,1},n,q,T)
#     function ∇scattered(δ)
#         println("empty",typeof(δ))
#         return (NO_FIELDS, )
#     end
#     ScatteredOut(m,k,fixDim,n,q,T), ∇scattered
# end

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
