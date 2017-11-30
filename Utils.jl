# This is a version of convolution that is optimized for short lengths as well as long ones. The character is created by \star so ⋆
function ⋆{T}(a::Array{T}, b::Array{T})
    n = length(a)
    m = length(b)
    if max(n,m)<400
        if n > m
            return filt(a, 1, [b; zeros(n-1)])
        else
            return filt(b, 1, [a; zeros(m-1)])
        end
    else
        return conv(a,b)
    end
end
