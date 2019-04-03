"""
  nonlinearity is the supertype for the different types of nonlinearity
"""
abstract type nonlinearity end
struct absType <: nonlinearity end
struct ReLUType <: nonlinearity end
struct tanhType <: nonlinearity end
struct softplusType <: nonlinearity end
ReLU(x::R) where R<:Real = max(x,0)
ReLU(x::C) where C<:Complex = max(real(x),0)+im*max(imag(x),0)
logabs(x)=log.(abs.(x))

"""
    Tanh(x::Complex) = tanh(real(x))+im*tanh(imag(x))
"""
Tanh(x::T) where T<:Complex = tanh(real(x))+im*tanh(imag(x)) # note that this is not the analytic function tanh
Tanh(x::T) where T<:Real = tanh(x)
"""
    softplus(x::Real)
    softplus(x::Complex)
Not the standard softplus, but one biased so that we return zero at zero. The complex operates on each component separately instead of using complex log/exponential.
"""
softplus(x::T) where T<:Real = log(1+exp(x))-log(2)
softplus(x::T) where T<:Complex= log(1+exp(real(x)))-log(2) + im*(log(1+exp(imag(x)))-log(2))
# if you add methods, make sure you add them to this list, otherwise shattering won't be defined using them. Also add them to the export list, otherwise they won't be externally callable
# inverse of the softplus function (note that it isn't accurate for input smaller than -36 because of numerical issues in the softplus function itself)
spInverse(x::Real) = x>0 ? log(exp(x)-1) : -36.737
spInverse(x::Complex) = spInverse(real(x)) + im*spInverse(imag(x))
aTanh(x::Complex) = atanh(real(x)) + im*atanh(imag(x))

functionTypeTuple = [(abs,absType),(ReLU,ReLUType), (Tanh,tanhType), (softplus, softplusType)]
