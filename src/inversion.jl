"""
    pseudoInversion!(fullySheared::scattered{T,N}, layers::layeredTransform, nonlin::S) where {T, S<:nonlinearity}
given a scattered type `fullySheared`, a layeredTransform `layers` and a
nonlinearity `nonlin` that were used to create `fullySheared`, recreate the
input in `fullySheared.data` as best as is possible based off of only the
values in the output. This is done iteratively
"""
function pseudoInversion!(fullySheared::scattered{T,N},
                          layers::layeredTransform, nonlin::S) where {T, N,
                                                                      S<:nonlinearity}
    # point of order: we need fft plans
    fftPlans = createFFTPlans(layers, [size(dat) for dat in fullySheared.data],
                              iscomplex= (T<:Complex))

    # start by upsampling all of the outputs and storing them in the data in
    # all but the last layer
    for l = layers.m:-1:1
        resampleLayers!(fullySheared.data[l], fullySheared.output[l], nonlin)
    end

    # once that's done, we start at the deepest layers and work our way up. In
    # the deepest layer, we have a lossy estimate of the data, since we have no
    # information about the higher frequencies at this layer; additionally, we
    # have a final nonlinearity to undo
    estimateLastLayer!(fullySheared.data[end], fullySheared.output[end],
                       layers.shears[end], nonlin, fftPlans[end])

    for m=layers.m:-1:1
        estimateMidLayer!(m, fullySheared.data, layers.shears[m], nonlin,
                          fftPlans[m], T)
    end

    return fullySheared
end

function resampleLayers!(layerData, layerOutput, nonlin)
    innerAxis = axes(layerData)[(end-2):(end-1)]
    innerOutputAxis = axes(layerOutput)[(end-2):(end-1)]
    outerAxis = axes(layerData)[1:(end-3)]
    for outer in eachindex(view(layerOutput, outerAxis..., 1, 1,1))
        for i= 1:size(layerOutput)[end]
            layerData[outer, innerAxis..., i] =
                resample(layerOutput[outer, innerOutputAxis..., i], 0f0,
                         newSize = size(layerData)[(end-2):(end-1)])
        end
    end 
end


function estimateLastLayer!(layerData, layerOutput, shear, nonlin, P)
    # once we've upsampled, convolve with the dual frame-- this is not enough for recovery unless the support of the signal is strictly within the support of the averaging filter
    padBy = getPadBy(shear)
    innerAxis = axes(layerData)[(end-2):(end-1)]
    innerAxisOut = axes(layerOutput)[(end-2):(end-1)]
    outerAxis = axes(layerData)[1:(end-3)]
    for outer in eachindex(view(layerData, outerAxis..., 1, 1,1))
        for i=1:size(layerData)[end]
            reconstructedData = layerOutput[outer, innerAxisOut..., i]
            reconstructedData = resample(reconstructedData, 0f0,
                                         newSize =
                                         size(layerData)[(end-2):(end-1)])
            reconstructedData = reshape(reconstructedData,
                                        (size(reconstructedData)..., 1))
            layerData[outer, innerAxis...,i] =
                shearrec2D(reconstructedData, shear, P, true, padBy,
                           size(layerData[outer, innerAxis...,i]),
                           averaging=true)
        end
    end
    #println("estimate Last Layer, max(layerData) = $(maximum(layerData))")
end

function estimateMidLayer!(m, layerData, shear, nonlin, P, T)
    innerAxis = axes(layerData[m])[(end-2):(end-1)]
    innerNextAxis = axes(layerData[m+1])[(end-2):(end-1)]
    outerAxis = axes(layerData[m])[1:(end-3)]
    padBy = getPadBy(shear)
    for outer in eachindex(view(layerData, outerAxis..., 1, 1,1))
        for indexInData=1:size(layerData[m])[end]
            #println("m = $(m)")
            fromNextLayerSub = inverseNonlin(layerData[m+1][outer,
                                                            innerNextAxis...,
                                                            getChildren(shear,
                                                                        m,
                                                                        indexInData)],
                                             nonlin) 
            fromNextLayer = zeros(T, innerAxis..., length(getChildren(shear,
                                                                      m,
                                                                      indexInData)))
            for k =1:size(fromNextLayerSub,3)
                fromNextLayer[:, :, k] = resample(fromNextLayerSub[:,:,1], 0f0,
                                                  newSize =
                                                  size(fromNextLayer)[1:2])
            end
            postShearlet = cat(fromNextLayer, layerData[m][outer, innerAxis...,
                                                           indexInData], dims=3)
            layerData[m][outer, innerAxis..., indexInData] =
                shearrec2D(postShearlet, shear, P, true, padBy,
                           size(layerData[m][outer, innerAxis..., indexInData]))
        end
    end
    # println("estimate mid layer, m=$(m), max(layerData) = $(maximum(layerData[m]))")
end
