ReLU(x::R) where R <: Real = max(x, 0)
ReLU(x::C) where C <: Complex = max(real(x), 0) + im * max(imag(x), 0)
logabs(x) = log.(abs.(x))

"""
    Tanh(x::Complex) = tanh(real(x))+im*tanh(imag(x))
"""
Tanh(x::T) where T <: Complex = tanh(real(x)) + im * tanh(imag(x)) # note that this is not the analytic function tanh
Tanh(x::T) where T <: Real = tanh(x)
"""
    softplus(x::Real)
    softplus(x::Complex)
Not the standard softplus, but one biased so that we return zero at zero. The complex operates on each component separately instead of using complex log/exponential.
"""
softplus(x::T) where T <: Real = log(1 + exp(x)) - log(2)
softplus(x::T) where T <: Complex = log(1 + exp(real(x))) - log(2) + im * (log(1 + exp(imag(x))) - log(2))
piecewiseLinear(x::T) where T <: Real = (x >= 0 ? x : x / 2)
piecewiseLinear(x::T) where T <: Complex = ((real(x) >= 0 && imag(x >= 0)) ? x : x / 2)

"""
    spInverse(x::Union{<:Real, <:Complex})
inverse of the softplus function (note that it isn't accurate for input smaller
than -36 because of numerical issues in the softplus function itself).
"""
spInverse(x::Real) = x > 0 ? log(exp(x) - 1) : -36.737
spInverse(x::Complex) = spInverse(real(x)) + im * spInverse(imag(x))
aTanh(x::T) where T <: Real = abs.(x) < 1 ? atanh(real(x)) : sign(x) * 10
aTanh(x::Complex) = aTanh(real(x)) + im * aTanh(imag(x))
plInverse(x::Real) = (x > 0 ? x : 2 * x)
