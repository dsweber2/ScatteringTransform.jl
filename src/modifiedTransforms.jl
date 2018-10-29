using Wavelets, FFTW, SpecialFunctions, LinearAlgebra
import Wavelets.cwt
import Wavelets.wavelet
import Wavelets.eltypes
# TODO: someday, make this more efficient (precompute the filters, for example)
wavelet(WT.morl,3.5)
# define a type for averaging continuous wavelets
struct CFWA{T} <: ContinuousWavelet{T}
    scalingFactor  ::Float64 # the number of wavelets per octave, ie the scaling is s=2^(j/scalingfactor)
    fourierFactor  ::Float64
    coi            ::Float64
    α              ::Int   # the order for a Paul and the number of derivatives for a DOG
    σ              ::Array{Float64} # the morlet wavelet parameters (σ,κσ,cσ). NaN if not morlet.
    name           ::String
    # function CFW{T}(scalingfactor, fourierfactor, coi, daughterfunc, name) where T<: WaveletBoundary
        # new(scalingfactor, fourierfactor, coi, daughterfunc, name)
    # end
    averagingLength::Int # the number of scales to override with averaging
    averagingType  ::Symbol # either Dirac or mother; the first uniformly represents the lowest frequency information, while the second constructs a wavelet using Daughter that has mean frequency zero, and std equal to the first non-removed wavelet's mean

end
function eltypes(::CFWA{T}) where T
    T
end






function CFWA(wave::WC, scalingfactor::S=8, averagingType::Symbol=:Mother, boundary::T=WT.DEFAULT_BOUNDARY, averagingLength::Int=floor(Int,scalingfactor/2)) where {WC<:WT.WaveletClass, T<:WT.WaveletBoundary, S<:Real}
    if scalingfactor<=0
        error("scaling factor must be positive")
    end
    @assert (averagingType==:Mother ||averagingType==:Dirac)
    namee = WT.name(wave)[1:3]
    tdef = get(WT.CONT_DEFS, namee, nothing)
    tdef == nothing && error("transform definition not found; you gave $(namee)")
    # do some substitution of model parameters
    if namee=="mor"
        tdef = [eval(Meta.parse(replace(tdef[1],"σ" => wave.σ))), eval(Meta.parse(tdef[2])), -1, [wave.σ,wave.κσ,wave.cσ], WT.name(wave)]
    elseif namee[1:3]=="dog" || namee[1:3]=="pau"
        tdef = [eval(Meta.parse(replace(tdef[1], "α" => WT.order(wave)))), eval(Meta.parse(replace(tdef[2], "α"=> WT.order(wave)))), WT.order(wave), [NaN], WT.name(wave)]
    else
        error("I'm not sure how you got here. Apparently the WaveletClass you gave doesn't have a name. Sorry about that")
    end
    return CFWA{T}(Float64(scalingfactor), tdef...,averagingLength, averagingType)
end


function CFWA(wave::WC; scalingfactor::S=8, averagingType::Symbol=:Mother, boundary::T=WT.DEFAULT_BOUNDARY, averagingLength::Int=floor(Int,scalingfactor/2)) where {WC<:WT.WaveletClass, T<:WT.WaveletBoundary, S<:Real}
    return CFWA(wave, scalingfactor, averagingType, boundary, averagingLength)
end
# If you know the averaging length
function wavelet(c::W, s::S, averagingLength::T, averagingType::Symbol=:Mother, boundary::WT.WaveletBoundary=WT.DEFAULT_BOUNDARY) where {W<:WT.ContinuousWaveletClass, S<:Real,T<:Real}
    CFWA(c, s, averagingType, boundary, averagingLength)
end
# If you want a default averaging length, just input the type
function wavelet(c::W, s::S, averagingType::Symbol, boundary::WT.WaveletBoundary=WT.DEFAULT_BOUNDARY) where {W<:WT.ContinuousWaveletClass, S<:Real}
    CFWA(c, s, averagingType, boundary)
end





"""
    totalN = numScales(c::CFWA,n::S) where S<:Integer
given a CFWA structure and the size of a vector it is acting on, return the number of transformed elements, including the averaging layer
"""
function numScales(c::CFWA, n::S) where S<:Integer

    J1=floor(Int,(log2(n)-2)*c.scalingFactor)
    #....construct time series to analyze, pad if necessary
    if eltypes(c) == WT.padded
        base2 = round(Int,log(n)/log(2));   # power of 2 nearest to N
        n = n+2^(base2+1)-n
    elseif eltypes(c) == WT.DEFAULT_BOUNDARY
        n= 2*n
    else
        n=n
    end

    return J1+2-c.averagingLength
end



name(s::CFW) = s.name



"""
    daughter = Daughter(this::CFW, s::Real, ω::Array{Float64,1})

given a CFW object, return a rescaled version of the mother wavelet, in the fourier domain. ω is the frequency, which is fftshift-ed. s is the scale variable
"""
function Daughter(this::CFWA, s::Real, ω::Array{Float64,1})
    if this.name=="morl"
        daughter = this.σ[3]*(π)^(1/4)*(exp.(-(this.σ[1].-ω/s).^2/2).-this.σ[2]*exp.(-1/2*(ω/s).^2))
    elseif this.name[1:3]=="dog"
        daughter = normalize(im^(this.α)*sqrt(gamma((this.α)+1/2))*(ω/s).^(this.α).*exp.(-(ω/s).^2/2))
    elseif this.name[1:4]=="paul"
        daughter = zeros(length(ω))
        daughter[ω.>=0]=(2^this.α)/sqrt((this.α)*gamma(2*(this.α)))*((ω[ω.>=0]/s).^(this.α).*exp.(-(ω[ω.>=0]/s)))
    end
    return daughter
end




function findAveraging(c::CFWA{W}, ω::Vector{Float64}) where W<:WT.WaveletBoundary
    s = 2^(c.averagingLength/c.scalingFactor)
    if c.name=="morl"
        # for the Morlet wavelet, the distribution is just a Gaussian, so it has variance 1/s^2 and mean σ[1]*s
        # set the variance so that the averaging function has 1σ at the central frequency of the last scale
        if c.averagingType==:Mother
            s0 = c.σ[1]*s/3
            averaging = Daughter(c,s0, ω .+ c.σ[1]*s0)
        elseif c.averagingType==:Dirac
            # set the averaging window to take everything below the mean of the last wavelet equally
            averaging = zeros(Float64, size(ω))
            averaging[abs.(ω).<=c.σ[1]*s] .= 1
        end
    elseif c.name[1:4]=="paul"
        # It's a easy calculation to see that the mean of a paul wavelet of order m is (m+1)/s, while σ=sqrt(m+1)/s
        # set the variance so that the averaging function has 1σ at the central frequency of the last scale
        if c.averagingType==:Mother
            s0 = s*sqrt(c.α+1)
            averaging = Daughter(c,s0, ω.+(c.α .+ 1)*s0)
        elseif c.averagingType==:Dirac
            # set the averaging window to take everything below the mean of the last wavelet equally
            averaging = zeros(Float64, size(ω))
            averaging[abs.(ω).<=(c.α+1)*s] .= 1
        end
    elseif c.name[1:3]=="dog"
        if c.averagingType==:Mother
            # the derivative of a Gaussian has a pretty nasty form for the mean and variance; eventually, if you set σ_{averaging}=⟨ω⟩_{highest scale wavelet}, you will get the scale of the averaging function to be
            s0 = s*gamma((c.α+2)/2)/sqrt(gamma((c.α+1)/2)*(gamma((c.α+3)/2)-gamma((c.α+2)/2)))
            μ = sqrt(2)*s0*gamma((c.α+2)/2)/gamma((c.α+1)/2)
            averaging = Daughter(c,s0, ω.+μ)
        elseif c.averagingType==:Dirac
            # set the averaging window to take everything below the mean of the last wavelet equally
            averaging = zeros(Float64, size(ω))
            averaging[abs.(ω).<=sqrt(2)*s*gamma((c.α+2)/2)/gamma((c.α+1)/2)] .= 1
        end
    else
        error("$(c.name) hasn't been defined")
    end
    return averaging
end






"""
   wave = cwt(Y::AbstractArray{T}, c::CFWA{W}, averagingLength::Int; J1::S=NaN, averagingType::Symbol=:Mother) where {T<:Number, S<:Real, W<:WT.WaveletBoundary}

return the continuous wavelet transform along the final axis with averaging wave, which is (previous dimensions)×(nscales+1)×(signalLength), of type T of Y. The extra parameter averagingLength defines the number of scales of the standard cwt that are replaced by an averaging function. This has form averagingType, which can be one of :Mother or :Dirac- in the :Mother case, it uses the same form as for the daughters, while the dirac uses a constant. J1 is the total number of scales; default (when J1=NaN, or is negative) is just under the maximum possible number, i.e. the log base 2 of the length of the signal, times the number of wavelets per octave. If you have sampling information, you will need to scale wave by δt^(1/2).
"""
function cwt(Y::AbstractArray{T}, c::CFWA{W}, daughters::Array{ComplexF64,2}; J1::S=NaN, backOffset::Int=0) where {T<:Number, S<:Real, W<:WT.WaveletBoundary}
    n1 = size(Y)[end];
    # J1 is the total number of elements
    if (isnan(J1) || (J1<0)) && c.name!="morl"
        J1=floor(Int,(log2(n1)-2)*c.scalingFactor)-backOffset
    elseif isnan(J1) || (J1<0)
        J1=floor(Int,(log2(n1)-2)*c.scalingFactor)-backOffset
    end

    #....construct time series to analyze, pad if necessary
    if eltypes(c) == WT.padded
        base2 = round(Int,log(n1)/log(2));   # power of 2 nearest to N
        x = [Y; zeros(2^(base2+1)-n1)];
    elseif eltypes(c) == WT.DEFAULT_BOUNDARY
        x = [Y; reverse(Y,length(size(Y)))]
    else
        x= Y
    end

    n = size(x)[end]
    x̂ = FFTW.fft(x, length(size(x)));    # [Eqn(3)]

    #....construct wavenumber array used in transform [Eqn(5)]


    # If the vector isn't long enough to actually have any other scales, just return the averaging
    if J1+2-c.averagingLength<=0 || J1==0
        wave = zeros(Complex{Float64}, size(Y)..., 1)
        mother = daughters[:,1]
        for i in eachindex(view(x̂, axes(x̂)[1:end-1]..., 1))
            wave[i,:]= x̂[i,:].* mother
        end
        return ifft(wave, length(size(wave))-1)
    end
    # TODO: switch J1 and n, and check everywhere this breaks
    wave = zeros(Complex{Float64}, size(Y)..., J1+2-c.averagingLength);  # define the wavelet array
    # loop through all scales and compute transform
    for a1 in c.averagingLength:J1
        daughter = daughters[:,a1-c.averagingLength + 2]
        for i in eachindex(view(x̂, axes(x̂)[1:end-1]..., 1))
            wave[i, :, a1-c.averagingLength + 2] = x̂[i,:].*daughter  # wavelet transform[Eqn(4)]
        end
    end
    for i in eachindex(view(x̂, axes(x̂)[1:end-1]..., 1))
        wave[i, :, 1] = x̂[i,:].*daughters[:,1]
    end
    wave = ifft(wave, length(size(wave))-1)
    ax = axes(wave)
    if length(ax)>2
        wave = wave[ax[1:end-2]..., 1:n1, ax[end]]  # get rid of padding before returning
    else
        wave = wave[1:n1, ax[end]]  # get rid of padding before returning
    end

    return wave
end
#TODO: include some logic about the types checking whether T is complex, and then using that as the base type (for both copies of wave)


function cwt(Y::AbstractArray{T}, c::CFWA{W}; J1::S=NaN, backOffset::Int=0) where {T<:Number, S<:Real, W<:WT.WaveletBoundary}
    daughters = computeWavelets(Y, c; J1=J1, backOffset=backOffset)
    return cwt(Y, c, daughters; J1=J1, backOffset=backOffset)
end

"""
    just precomputes the wavelets used by transform c::CFWA{W}. For details, see cwt
"""
function computeWavelets(Y::AbstractArray{T}, c::CFWA{W}; J1::S=NaN, backOffset::Int=0) where {T<:Number, S<:Real, W<:WT.WaveletBoundary}
    n1 = size(Y)[end]
    # J1 is the total number of elements
    if (isnan(J1) || (J1<0)) && c.name!="morl"
        J1=floor(Int,(log2(n1)-2)*c.scalingFactor)-backOffset
    elseif isnan(J1) || (J1<0)
        J1=floor(Int,(log2(n1)-2)*c.scalingFactor)-backOffset
    end
    #....construct time series to analyze, pad if necessary
    if eltypes(c) == WT.padded
        base2 = round(Int,log(n1)/log(2));   # power of 2 nearest to N
        n= size(Y)[end]+2^(base2+1)-n1
    elseif eltypes(c) == WT.DEFAULT_BOUNDARY
        n = 2*size(Y)[end]
    else
        n=size(Y)[end]
    end
    ω = [0:ceil(Int, n/2); -floor(Int,n/2)+1:-1]*2π

    if J1+2-c.averagingLength<=0 || J1==0
        mother = zeros(Complex{Float64}, size(Y)[end], 1)
        mother[:,1] = findAveraging(c,ω)
        return mother
    end

    daughters = zeros(Complex{Float64}, n, J1+2-c.averagingLength)
    for a1 in c.averagingLength:J1
        daughters[:, a1-c.averagingLength + 2] = Daughter(c, 2.0^(a1/c.scalingFactor), ω)
    end
    daughters[:,1] = findAveraging(c,ω)
    return daughters
end

"""
    daughters, ω = getScales(n1, c::CFW{W}, averagingLength::Int; J1::S=NaN, averagingType::Symbol=:Mother) where {T<:Real, S<:Real, W<:WT.WaveletBoundary}

    return the wavelets and the averaging functions used as row vectors in the Fourier domain. The averaging function is first, and the frequency variable is returned as ω. n1 is the length of the input
"""
function getScales(n1::Int, c::CFWA{W}; J1::S=NaN) where {S<:Real, W<:WT.WaveletBoundary}
    # J1 is the total number of elements
    if (isnan(J1) || (J1<0)) && c.name!="morl"
        J1=floor(Int,(log2(n1))*c.scalingFactor);
    elseif isnan(J1) || (J1<0)
        J1=floor(Int,(log2(n1)-2)*c.scalingFactor);
    end
    #....construct time series to analyze, pad if necessary
    if eltypes(c) == WT.padded
        base2 = round(Int,log(n1)/log(2));   # power of 2 nearest to N
        n = n1+2^(base2+1)-n1
    elseif eltypes(c) == WT.DEFAULT_BOUNDARY
        n= 2*n1
    else
        n=n1
    end

    #....construct wavenumber array used in transform [Eqn(5)]
    ω = [0:ceil(Int, n/2); -floor(Int,n/2)+1:-1]*2π

    daughters = Array{Complex{Float64},2}(J1+2-c.averagingLength,n)
    # loop through all scales and compute transform
    for a1 in c.averagingLength:J1
        daughter = Daughter(c, 2.0^(a1/c.scalingFactor), ω)
        daughters[a1-c.averagingLength+2,:] = daughter
    end
    daughters[1,:] = findAveraging(c,ω)

    return (daughters,ω)
end

getScales(c::CFWA{W}, n1::Int; J1::S=NaN) where {S<:Real, W<:WT.WaveletBoundary} = getScales(n1, c; J1=J1)
