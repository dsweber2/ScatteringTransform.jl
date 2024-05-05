function monogenic_filter(RHO; Monotype = GaussianLP())
    
    if typeof(Monotype) == GaussianLP
        # For Guassian Low Pass Filter
        LP  =  exp.( -(RHO.^2)/8 );       # Gaussian low-pass filter
        HP  =  sqrt.( 1 .- LP.^2 );          # Complementary high-pass filter
        
    elseif typeof(Monotype) == GaussianHP
        # For Guassian High Pass Filter
        HP  =  1 .- exp.( -(RHO.^2)/2 ); # Gaussian-based high-pass filter
        LP  =  sqrt.( 1 .- HP.^2 );      # Complementary low-pass
    end
    
    return LP, HP
    
end


# Generates 2D Fourier polar coordinates

#function meshgrid(x, y)
#    X = [i for i in x, j in 1:length(y)];
#    Y = [j for i in 1:length(x), j in y];
#    return X, Y
#end

function FFT_radial(siz)
# Creates an MxN grid of Fourier coordinates 
# - First sample is the "DC coordinate" (no 'fftshift' done)
# - Frequencies span [-pi;pi]^2

# 'siz' : size in the form '[M,N]'
# 'RHO' : absolute frequency or radial frequency (distance from DC)
# 'RZ'  : 'orientation', embedded in a complex exponential
# 'RZ' turns out to coincide with the Riesz-transform frequency response.
# The singular value of riez at DC is set to 1 for convenience
# (should be 0 in theory).
    M = siz[1];
    N = siz[2];
    W1 = (-floor(M/2):ceil(M/2).-1)/M * ones(1, N);
    W2 = ones(M,1)*((-floor(N/2):ceil(N/2).-1)/N)';
    #W1,W2 = meshgrid(  (-floor(M/2):ceil(M/2).-1)/M  , (-floor(N/2):ceil(N/2).-1)/N  );
    RHO  = ifftshift( 2*pi*sqrt.( W1.^2 + W2.^2 )  ); # Radial freq. coord.
    RZ = ifftshift(  -1im * exp.(  1im * atan.(W2,W1)  )    ); # Riesz filter
    return RHO, RZ
end



# similar to shearingLayer in ST

# X_Size and Y_Size should be equal

#For filters
    # First filter: Rz
    # Second filter: RHO [2:3:end]
    # Third filter: HP  [3:3:end] 
    # Fourth filter: LP [4:3:end]


# similar to shearingLayer in ST
# X_Size and Y_Size should be equal

function MonogenicLayer(inputSize::Union{Int,NTuple{N, T}}; scale = 4, 
                       boundary=FourierFilterFlux.Periodic(), 
                       init = Flux.glorot_normal, dType = Float32, σ = abs,
                       trainable = false, plan = true,
                       averagingLayer = false, Monotype = GaussianLP()) where {N,T}

    #scale = defaultShearletScale(inputSize, scale)
    
    # Compute the 2D Fourier polar coordinates
    RHO_,RZ = FFT_radial(  inputSize[1:2]  );
    
    monogenic = cat(RZ, dims = 3);
    
    # without pyramid structure:
    for j = 1:scale
        RHO = RHO_ .* (2^(j-1));
        
        LP, HP = monogenic_filter(RHO; Monotype = Monotype);
        
        monogenic = cat(monogenic, RHO, HP, LP, dims = 3);
    end


    if dType <: Real
        monogenic = Complex{dType}.(monogenic) # correct the element type
    else
        monogenic = dType.(monogenic)
    end


    return MonoConvFFT(monogenic, inputSize, σ, plan=plan, boundary=boundary, dType = dType,
                   trainable=trainable, OT = dType, scale = scale, averagingLayer = averagingLayer)
end






