import Wavelets.cwt
import Wavelets.wavelet
import Wavelets.eltypes
import Wavelets.WT
import Shearlab.Shearletsystem2D

# define a type for averaging continuous wavelets
struct CFWA{T} <: ContinuousWavelet{T}
    scalingFactor  ::Float64 # the number of wavelets per octave, ie the scaling is s=2^(j/scalingFactor)
    fourierFactor  ::Float64
    coi            ::Float64
    α              ::Int   # the order for a Paul and the number of derivatives for a DOG
    σ              ::Array{Float64} # the morlet wavelet parameters (σ,κσ,cσ). NaN if not morlet.
    name           ::String
    # function CFW{T}(scalingFactor, fourierfactor, coi, daughterfunc, name) where T<: WaveletBoundary
    # new(scalingFactor, fourierfactor, coi, daughterfunc, name)
    # end
    averagingLength::Int # the number of scales to override with averaging
    averagingType  ::Symbol # either Dirac or mother; the first uniformly represents the lowest frequency information, while the second constructs a wavelet using Daughter that has mean frequency zero, and std equal to the first non-removed wavelet's mean
    frameBound     ::Float64 # if positive, set the frame bound of the transform to be frameBound. Otherwise leave it so that each wavelet has an L2 norm of 1
end
function eltypes(::CFWA{T}) where T
    T
end






function CFWA(wave::WC, scalingFactor::S=8.0, averagingType::Symbol=:Mother, boundary::T=WT.DEFAULT_BOUNDARY, averagingLength::Int=floor(Int,scalingFactor/2), frameBound::Float64=1.0) where {WC<:WT.WaveletClass, T<:WT.WaveletBoundary, S<:Real}
  @assert scalingFactor > 0
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
  return CFWA{T}(Float64(scalingFactor), tdef...,averagingLength, averagingType, frameBound)
end

# If you know the averaging length
function wavelet(c::W, s::S, averagingLength::T, averagingType::Symbol=:Mother, boundary::WT.WaveletBoundary=WT.DEFAULT_BOUNDARY, frameBound::Float64=-1.0) where {W<:WT.ContinuousWaveletClass, S<:Real,T<:Real}
  CFWA(c, s, averagingType, boundary, averagingLength)
end
# If you want a default averaging length, just input the type
function wavelet(c::W, s::S, averagingType::Symbol, boundary::WT.WaveletBoundary=WT.DEFAULT_BOUNDARY, frameBound::Float64=-1.0) where {W<:WT.ContinuousWaveletClass, S<:Real}
  CFWA(c, s, averagingType, boundary; frameBound=frameBound)
end





@doc """
      totalN = numScales(c::CFWA,n::S; backOffset=0,nScales=-1) where S<:Integer
      totalN = numScales(c::Shearletsystem2D) where S<:Integer
  given a CFWA structure and the size of a vector it is acting on,
      return the number of transformed elements, including the
      averaging layer.
  alternatively, given a Shearletsystem2D structure, return the number
      of shearlets, including the averaging layer.
  """
function numScales(c::CFWA, n::S; backOffset=0,nScales=-1) where S<:Integer
    if isnan(nScales) || nScales<0
        nScales=floor(Int,(log2(max(n,1))-2)*c.scalingFactor)-backOffset-c.averagingLength
    end
    return nScales
end

function numScales(c::Shearletsystem2D) where S<:Integer
    return size(c.shearletIdxs,1)
end

function getJ1(c,nScales, backOffset, n1)
    if nScales<0 || isnan(nScales)
        nScales = numScales(c, n1; backOffset=0)
    end
    J1= nScales+c.averagingLength-1
    return J1
end



name(s::CFW) = s.name



@doc """
      daughter = Daughter(this::CFW, s::Real, ω::Array{Float64,1})

  given a CFW object, return a rescaled version of the mother wavelet, in the fourier domain. ω is the frequency, which is fftshift-ed. s is the scale variable. This extension just allows us to use a ::CFWA type instead of a CFW type.
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






@doc """
     wave = cwt(Y::AbstractArray{T}, c::CFWA{W}, averagingLength::Int; J1::S=NaN, averagingType::Symbol=:Mother) where {T<:Number, S<:Real, W<:WT.WaveletBoundary}

  return the continuous wavelet transform along the final axis with averaging wave, which is (previous dimensions)×(nscales+1)×(signalLength), of type T of Y. The extra parameter averagingLength defines the number of scales of the standard cwt that are replaced by an averaging function. This has form averagingType, which can be one of :Mother or :Dirac- in the :Mother case, it uses the same form as for the daughters, while the dirac uses a constant. J1 is the total number of scales; default (when J1=NaN, or is negative) is just under the maximum possible number, i.e. the log base 2 of the length of the signal, times the number of wavelets per octave. If you have sampling information, you will need to scale wave by δt^(1/2).
  """
function cwt(Y::AbstractArray{T,N}, c::CFWA{W}, daughters::Array{T, M},
             fftPlan::FFTW.rFFTWPlan{T,A,B,C} = plan_rfft([1]); nScales::S =
             NaN, backOffset::Int=0) where {T<:Real, S<:Real, U<:Number,
                                            W<:WT.WaveletBoundary, N, M,A,B,C}
    # TODO: complex input version of this
    @assert typeof(N)<:Integer
    @assert typeof(M)<:Integer
    @assert M==1 || M==2
    if N==1
        Y= reshape(Y,(1,size(Y)...))
    end
    # TODO: version of this with a preplanned fft
    n1 = size(Y)[end];
    # J1 is the final scale
    J1 = getJ1(c,nScales, backOffset, n1)
    if (isnan(nScales) || (nScales<0)) && c.name!="morl"
        J1=floor(Int,(log2(n1)-2)*c.scalingFactor)-backOffset
    elseif nScales>0
        J1 = nScales+c.averagingLength-1
    end
    
    #....construct time series to analyze, pad if necessary
    if eltypes(c)() == WT.padded
        base2 = round(Int,log(n1)/log(2));   # power of 2 nearest to N
        x = [Y zeros(2^(base2+1)-n1)];
    elseif eltypes(c)() == WT.DEFAULT_BOUNDARY
        x = cat(Y, reverse(Y,dims = length(size(Y))), dims = length(size(Y)))
    else
        x= Y
    end

    # check if the plan we were given is a dummy or not
    if size(fftPlan)==(1,)
        fftPlan = plan_rfft(x[[1 for i=1:length(size(x))-1]..., :])
    end
    n = size(x)[end]
    x̂ = zeros(Complex{eltype(x)}, size(x)[1:end-1]..., div(size(x)[end],2))
    x̂ = fftPlan * x    # [Eqn(3)]
    
    reshapeSize = (ones(Int, length(size(x))-1)..., size(daughters,1))
    #....construct wavenumber array used in transform [Eqn(5)]
    # If the vector isn't long enough to actually have any other scales, just
    # return the averaging
    if J1+1-c.averagingLength<=1
        wave = zeros(T, size(x)..., 1)
        mother = reshape(daughters[:, 1], reshapeSize)
        wave = fftPlan \ (x̂ .* mother)
        return wave
    end
    # TODO: switch J1 and n, and check everywhere this breaks
    wave = zeros(T, size(x)..., size(daughters)[end]);  # define the wavelet
    # array
    outer = axes(x)
    # loop through all scales and compute transform
    for j in 1:size(daughters,2)
        daughter = reshape(daughters[:, j], reshapeSize)
        wave[outer..., j] = fftPlan \ (x̂ .* daughter)  # wavelet transform
        # [Eqn(4)]
    end
    ax = axes(wave)
    if length(ax)>2
        wave = wave[ax[1:end-2]..., 1:n1, ax[end]]  # get rid of padding before
        # returning
    else
        wave = wave[1:n1, ax[end]]  # get rid of padding before returning
    end
    return wave
end

function cwt(Y::AbstractArray{T,N}, c::CFWA{W}, daughters::Array{U,2},
             fftPlan::Future; nScales::S=NaN, backOffset::Int=0) where {T<:Real,
                                                                        S<:Real, U<:Number,
                                                                        W<:WT.WaveletBoundary, N}
    pl = fetch(fftPlan)
    return cwt(Y, c, daughters, pl; nScales=nScales, backOffset=backOffset)
end

#TODO: include some logic about the types checking whether T is complex, and then using that as the base type (for both copies of wave)


function cwt(Y::AbstractArray{T}, c::CFWA{W}; nScales::S=NaN,
             backOffset::Int=0) where {T<:Number, S<:Real,
                                       W<:WT.WaveletBoundary}
    daughters = computeWavelets(Y, c; nScales=nScales, backOffset=backOffset)
    return cwt(Y, c, daughters; nScales=nScales, backOffset=backOffset)
end




@doc """
      computeWavelets(Y::AbstractArray{T}, c::CFWA{W}; J1::S=NaN, backOffset::Int=0) where {T<:Number, S<:Real, W<:WT.WaveletBoundary}
  just precomputes the wavelets used by transform c::CFWA{W}. For details, see cwt
  """
function computeWavelets(n1::Integer, c::CFWA{W}; nScales::S=NaN, backOffset::Int=0,T=Float64) where {S<:Real, W<:WT.WaveletBoundary}
    # J1 is the total number of elements
    J1 = getJ1(c, nScales, backOffset, n1)
    nScales = numScales(c, n1; backOffset=backOffset,nScales=nScales)
    #....construct time series to analyze, pad if necessary
    if eltypes(c)() == WT.padded
        base2 = round(Int,log(n1)/log(2));   # power of 2 nearest to n1
        n= 2^(base2+1)
    elseif eltypes(c)() == WT.DEFAULT_BOUNDARY
        n = 2*n1
    else
        n=n1
    end
    ω = [0:ceil(Int, n/2); -floor(Int,n/2)+1:-1]*2π
    
    if J1+1-c.averagingLength<=1 || J1==0
        mother = zeros(T, n1+1, 1)
        mother[:,1] = findAveraging(c,ω)[1:(n1+1)]
        return mother
    end

    daughters = zeros(T, n1+1, nScales+1)
    for a1 in (c.averagingLength-1):J1
        # since we use a real fft plan, we only need half of the coefficients
        daughters[:, a1-c.averagingLength + 2] = Daughter(c, 2.0^(a1/c.scalingFactor), ω)[1:(n1+1)]
    end
    daughters[:, 1] = findAveraging(c,ω)[1:(n1+1)]
    # normalize so energy is preserved
    daughters = daughters./sqrt.(sum(abs.(daughters).^2, dims=1))
    if c.frameBound > 0
        daughters = daughters.*(c.frameBound/norm(daughters,2))
    end
    return daughters
end
function computeWavelets(Y::AbstractArray{<:Integer}, c::CFWA{W}; nScales::S=NaN,
                backOffset::Int=0, T=Float64) where {S<:Real,
                                                     W<:WT.WaveletBoundary}
    return computeWavelets(size(Y)[end], c; nScales=nScales,
                           backOffset=backOffset, T=T)
end


@doc """
      daughters, ω = getScales(n1, c::CFW{W}, averagingLength::Int; J1::S=NaN, averagingType::Symbol=:Mother) where {T<:Real, S<:Real, W<:WT.WaveletBoundary}

  return the wavelets and the averaging functions used as row
  vectors in the Fourier domain. The averaging function is first, and
  the frequency variable is returned as ω. n1 is the length of the
  input
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

function getScales(c::CFWA{W}, n1::Int; J1::S=NaN) where {S<:Real,
                                                          W<:WT.WaveletBoundary}
    getScales(n1, c; J1=J1)
end






###################################################################################
#################### Shattering methods ###########################################
###################################################################################

function sheardec2D(X, shearletSystem, P::Future, padded, padBy)
    sheardec2D(X,shearletSystem, fetch(P), padded, padBy)
end

function sheardec2D(X::SubArray{Complex{T}, N},
                    shearletSystem::Shearlab.Shearletsystem2D{T},
                    P::FFTW.cFFTWPlan{Complex{T}, A, B, C},
                    padded::Bool, padBy::Tuple{Int, Int}) where {T <: Real, A,
                                                       B, C, N}
    if shearletSystem.gpu;
        coeffs = AFArray(zeros(Complex{T},
                    size(shearletSystem.shearlets))) 
    else
        coeffs = zeros(Complex{T}, size(shearletSystem.shearlets))
    end

    #compute shearlet coefficients at each scale
    #not that pointwise multiplication in the fourier domain equals convolution
    #in the time-domain
    for j = 1:shearletSystem.nShearlets
        if padded
            # The fourier transform of X
            Xfreq = fftshift(P*(ifftshift(pad(X, padBy))))
            coeffs[:,:,j] = real.(P \ ifftshift(Xfreq.*conj(shearletSystem.shearlets[:,:,j])))[padBy[1] .+ (1:size(X,1)), padBy[2] .+ (1:size(X,2))]
        else
            coeffs[:,:,j] = real.(fftshift(P \ ifftshift(Xfreq.*conj(shearletSystem.shearlets[:,:,j]))))
        end
    end
    return coeffs
end # sheardec2D

function sheardec2D(X::Array{Complex{T}, 2},
           shearletSystem::Shearlab.Shearletsystem2D{T},
           P::FFTW.cFFTWPlan{Complex{T}, A, B, C}, padded::Bool,
           padBy::Tuple{Int, Int}) where {T <: Real, A,
                                          B, C, N}
    sheardec2D(view(X, :, :), shearletSystem, P, padded, padBy)
end


function sheardec2D(X::SubArray{T,N},
                    shearletSystem::Shearlab.Shearletsystem2D{T},
                    P::FFTW.rFFTWPlan{T,B,C,D}, padded::Bool,
                    padBy::Tuple{Int, Int}) where {T<:Real, N, B, C, D}
    if shearletSystem.gpu;
        coeffs = AFArray(zeros(Complex{T},
                               size(shearletSystem.shearlets))) 
    else
        coeffs = zeros(T, size(X)..., size(shearletSystem.shearlets,3))
    end
    
    # compute shearlet coefficients at each scale
    # not that pointwise multiplication in the fourier domain equals convolution
    # in the time-domain
    used1 = padBy[1] .+ (1:size(X)[end-1])
    used2 = padBy[2] .+ (1:size(X)[end])
    for j = 1:shearletSystem.nShearlets
        # The fourier transform of X
        neededShear = conj(shearletSystem.shearlets[:, :, j])
        shearing!(X, neededShear, P,  coeffs, padBy, used1, used2, j)
    end
    return coeffs
end # sheardec2D


#################### Reconstruction main functions ##############################


function shearrec2D(coeffs::SubArray{T,N},
                    shearletSystem::Shearlab.Shearletsystem2D{T},
                    P::FFTW.rFFTWPlan{T,B,C,D}, padded::Bool,
                    padBy::Tuple{Int, Int}, resultSize::Tuple{Int,Int};
                    averaging::Bool=false) where {T<:Real, N, B, C, D}
    if N==3
        X = zeros(Complex{T}, size(shearletSystem.shearlets)[1:2])
    else
        X = zeros(Complex{T}, size(coeffs)[1:end-3]...,
                  size(shearletSystem.shearlets)[1:2]...)
    end
    
    # sum X in the Fourier domain over the coefficients at each scale/shearing
    used1 = padBy[1] .+ (1:resultSize[1])
    used2 = padBy[2] .+ (1:resultSize[2])
    if averaging
        neededShear = shearletSystem.shearlets[:,:, end]
        unshearing!(X, neededShear, P, coeffs, padBy, size(coeffs,3))
    else
        for j = 1:shearletSystem.nShearlets
            # The fourier transform of X
            neededShear = shearletSystem.shearlets[:, :, j]
            unshearing!(X, neededShear, P,  coeffs, padBy, j)
        end
    end
    realX = zeros(T, size(coeffs)[1:end-1])
    realX!(realX, X, shearletSystem.dualFrameWeights, P, used1, used2)
    return realX
end


function shearrec2D(coeffs, shearletSystem, P::Future, padded, padBy,
                    resultSize; averaging=false)
    shearrec2D(coeffs, shearletSystem, fetch(P), padded, padBy, resultSize,
               averaging=averaging)
end
    
function shearrec2D(coeffs::Array{T,N},
                    shearletSystem::Shearlab.Shearletsystem2D{T},
                    P::FFTW.rFFTWPlan{T,B,C,D}, padded::Bool,
                    padBy::Tuple{Int, Int}, resultSize; averaging::Bool=false) where {T<:Real, N, B, C, D}
    shearrec2D(view(coeffs, axes(coeffs)...), shearletSystem, P, padded, padBy,
               resultSize; averaging = averaging)
end
 




###################### Helper methods shearlets ###################################


function unshearing!(X, neededShear, P, coeffs, padBy, j) where {N,T}
    for i in eachindex(view(X, axes(X)[1:end-2]..., 1, 1))
        cFreq = fftshift( P * (pad(coeffs[i, :, :, j], padBy)))
        X[i, :, :] = X[i, :, :] + cFreq .* neededShear
    end
end

function unshearing!(X::AbstractArray{Complex{T},2}, neededShear, P,
                     coeffs::AbstractArray{T,3}, padBy, j) where {N,T}
    cFreq = fftshift( P * (pad(coeffs[:, :, j], padBy)))
    X[:, :] = X[:, :] + cFreq .* neededShear
end


function realX!(realX, X, duals, P, used1, used2)
    for i in eachindex(view(X, axes(X)[1:end-2]..., 1, 1))
        realX[i, :, :] = real.(fftshift((P \ (ifftshift(X[i,:,:] ./
                                                        (duals))))))[used1,
                                                                     used2]
    end
end
function realX!(realX, X::AbstractArray{T,2}, duals, P, used1, used2) where {T}
    realX[:, :] = real.(fftshift(P \ (ifftshift(X[:,:] ./ (duals)))))[used1,
                                                                      used2]
end


function shearing!(X::AbstractArray{T,N}, neededShear, P, coeffs, padBy, used1,
                   used2, j) where {N,T}
    for i in eachindex(view(X, axes(X)[1:end-2]..., 1, 1))
        Xfreq = fftshift( P * ifftshift(pad(X[i, :, :], padBy)))
        coeffs[i,:,:,j] = real.(P \ ifftshift(Xfreq .* neededShear))[used1,
                                                                         used2]
    end
end

function shearing!(X::AbstractArray{T,2}, neededShear, P, coeffs, padBy, used1,
                   used2, j) where T
    Xfreq =  fftshift( P * ifftshift(pad(X[:, :], padBy)))
    coeffs[:,:,j] = real.(P \ ifftshift(Xfreq .* neededShear))[used1,
                                                               used2]
end


sheardec2D(X::Array{T,2}, shearletSystem::Shearlab.Shearletsystem2D{T},
           P::FFTW.rFFTWPlan{T,B,C,D}, padded::Bool, padBy::Tuple{Int, Int}) where {T<:Real, N, B, C, D} = sheardec2D(view(X,:,:), shearletSystem, P, padded, padBy)


function getPadBy(shearletSystem::Shearlab.Shearletsystem2D{T}) where {T<:Number}
    padBy = (0, 0)
    for sysPad in shearletSystem.support
        padBy = (max(sysPad[1][2]-sysPad[1][1], padBy[1]), max(padBy[2], sysPad[2][2]-sysPad[2][1]))
    end
    return padBy
end


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
function averagingFunction(X::SubArray{Complex{T}, N},
                           shearletSystem::Shearlab.Shearletsystem2D{T},
                           P::FFTW.cFFTWPlan{Complex{T},A,B,C}, padded::Bool,
                           padBy::Tuple{Int, Int}) where{T <: Real, A, B, C, N} 
    averagingFunction(view(X,:,:), shearletSystem, P, padded, padBy, true)
end
function averagingFunction(X::SubArray{T, N},
                           shearletSystem::Shearlab.Shearletsystem2D{T},
                           P::FFTW.rFFTWPlan{T,A,B,C}, padded::Bool,
                           padBy::Tuple{Int,Int}) where{T <: Real, A, B, C, N}
    averagingFunction(X, shearletSystem, P, padded, padBy, true)
end



function averagingFunction(X, shearletSystem, P::Future, padded, padBy)
    return averagingFunction(X, shearletSystem, fetch(P), padded, padBy)
end

