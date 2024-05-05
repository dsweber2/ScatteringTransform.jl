# transform
# similar to (shears::ConvFFT)



function (Mono::MonoConvFFT)(x::AbstractArray{<:Number,4})

    # input: (nx, ny, nchannels, nexamples)
    # output: (nx, ny, nfilters, nchannels, nexamples)
    
    fftPlan = Mono.fftPlan;
    scale = Mono.scale;
    averagingLayer = Mono.averagingLayer;
    
    w = Mono.weight;
    # First filter: Rz
    # Second filter: RHO 
    # Third filter: HP  
    # Fourth filter: LP
     
    # cancel normalization to have "frequency responses in [0;1]"
    norm_csts = map(i -> 2 .^i ,[0:scale-1])[1];
    
    xbc, usedInds = applyBC(x, Mono.bc, ndims(w) - 1);
    if size(xbc) != size(fftPlan)
        xbc = reshape(xbc, size(fftPlan))
    end
    
    lfband = xbc;
    avg_out = 0;
    
    for scal=1:scale
        LF = fftPlan * lfband;
        lfband = real(fftPlan \ ( LF .* w[:,:,4 + 3 * (scal - 1)] ));
        HF = LF .* w[:,:,3 + 3 * (scal - 1)] ./ (norm_csts[scal]*sqrt(2));
        
        
        riez = fftPlan \ ( HF .* w[:,:,1] );           # Riesz transform
        
        out1 = real(fftPlan \ (HF) ); # Primary part
        out2 = real(riez);      # x-Riesz part
        out3 = imag(riez);      # y-Riesz part
        
        # get rid of the padding
        out1 = out1[usedInds..., axes(out1)[length(usedInds) + 1:end]...] 
        out2 = out2[usedInds..., axes(out2)[length(usedInds) + 1:end]...]
        out3 = out3[usedInds..., axes(out3)[length(usedInds) + 1:end]...]
        
        if scal == 1
            nextLayer = cat(out1, out2, out3, dims = 5);
            # average layer filtering
            avg_out = lfband[usedInds..., axes(lfband)[length(usedInds) + 1:end]...];
        else
            nextLayer = cat(nextLayer, out1, out2, out3, dims = 5);
        end


        if scal == scale
            nextLayer = cat(nextLayer, avg_out, dims = 5);
            if averagingLayer == true
                nextLayer = nextLayer[:,:,:,:,end:end];
            end
        end    

        lfband = lfband.*2; # normalization due to the undecimated setting

        

    end
    
    nextLayer = permutedims(nextLayer, (1,2,5,3,4));

    
    return Mono.σ.(nextLayer)
end



# similar to (shears::ConvFFT)
function (Mono::MonoConvFFT)(x::AbstractArray{<:Number,5})

# input: (nx, ny, nfilt, nchan, nexamples)
# output: (nx, ny, nfilters2, nfilters1, nchan, nexamples)
    
    fftPlan = Mono.fftPlan;
    scale = Mono.scale;
    averagingLayer = Mono.averagingLayer;
    
    w = Mono.weight;
    # First filter: Rz
    # Second filter: RHO 
    # Third filter: HP  
    # Fourth filter: LP
     
    # cancel normalization to have "frequency responses in [0;1]"
    norm_csts = map(i -> 2 .^i ,[0:scale-1])[1]
    
    xbc, usedInds = applyBC(x, Mono.bc, ndims(w) - 1);
    if size(xbc) != size(fftPlan)
        xbc = reshape(xbc, size(fftPlan))
    end
    
    lfband = xbc;
    avg_out = 0;
    
    for scal=1:scale
        LF = fftPlan * lfband;
        lfband = real(fftPlan \ ( LF .* w[:,:,4 + 3 * (scal - 1)] ));
        HF = LF .* w[:,:,3 + 3 * (scal - 1)] ./ (norm_csts[scal]*sqrt(2));
        
        
        riez = fftPlan \ ( HF .* w[:,:,1] );           # Riesz transform
        
        out1 = real(fftPlan \ (HF) ); # Primary part
        out2 = real(riez);      # x-Riesz part
        out3 = imag(riez);      # y-Riesz part
        
        # get rid of the padding
        out1 = out1[usedInds..., axes(out1)[length(usedInds) + 1:end]...] 
        out2 = out2[usedInds..., axes(out2)[length(usedInds) + 1:end]...]
        out3 = out3[usedInds..., axes(out3)[length(usedInds) + 1:end]...]
        
        if scal == 1
            nextLayer = cat(out1, out2, out3, dims = 6);
            avg_out = lfband[usedInds..., axes(lfband)[length(usedInds) + 1:end]...];
        else
            nextLayer = cat(nextLayer, out1, out2, out3, dims = 6);
        end

        

        if scal == scale
            nextLayer = cat(nextLayer, avg_out, dims = 6);
            if averagingLayer == true
                nextLayer = nextLayer[:,:,:,:,:,end:end];
            end
        end

        lfband = lfband.*2; # normalization due to the undecimated setting

    end
    
    nextLayer = permutedims(nextLayer, (1,2,6,3,4,5));
    
    return Mono.σ.(nextLayer)
end
