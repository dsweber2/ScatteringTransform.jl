function thinSt(X::Array{T}, layers::stParallel; nonlinear::Function = abs, subsam::Function = bspline, stType::String = "full", outputSubsample::Tuple{Int,Int} = (-1, 1), totalScales = [-1 for i = 1:layers.m+1]) where {T<:Real}
  @assert length(totalScales) == layers.m + 1
  # Insist that X has to have at least one extra meta-dimension, even if it is 1
  if length(size(X)) == length(size(layers.subsampling))
    X = reshape(X, (1, size(X)...))
  end
  numChildλ = 0
  #println("initial")
  k, n, q, dataSizes, outputSizes, resultingSize = calculateThinStSizes(layers, outputSubsample, size(X), totalScales = totalScales)
  #println("the required sizes are dataSizes = $(dataSizes)")
  #println("outputSizes = $(outputSizes)")
  nextData = [reshape(X, (size(X)..., 1)),]
  # create list of size references if we're subsampling the output an extra amount

  if outputSubsample[1] > 1 || outputSubsample[2] > 1
    concatOutput = SharedArray(zeros(Float64, outputSizes[1][1:end-k-1]..., sum(resultingSize .* q)))
  else
    concatOutput = SharedArray(zeros(Float64, outputSizes[1][1:end-k-1]..., sum([prod(x[end-k:end]) for x in outputSizes])...))
  end
  # keep track of where we are in each meta dimension
  outPos = ones(Int64, outputSizes[1][1:end-k-1]...)
  nScalesLayers = [numScales(layers.shears[i], max(dataSizes[i][1] - 1, 1)) for i = 1:length(layers.shears)]
  for (i, layer) in enumerate(layers.shears[1:layers.m])
    #println("starting the actual transformation in layer $(i)")
    cur = nextData #data from the previous layer
    #println("size(cur) = $(size(cur[1]))")
    # only store the intermediate results in intermediate layers
    if i < layers.m
      # store the data in a channel, along with the number of it's parent
      dataChannel = Channel{Tuple{Array{length(dataSizes[i+1])},Int64}}(size(cur)[end])
      nextData = Array{SharedArray{Float64,length(dataSizes[i+1])},1}(undef, size(cur[1])[end] * length(cur))
    end
    # if we're decreasing, we need to keep track of what path the current index corresponds to
    if stType == "decreasing" && i >= 2
      keeper = [1 for k = 1:i-1]
    elseif stType == "decreasing" && i == 1
      keeper = [1 for k = 1:i]
    end

    ######################################################################################################################################################################################################################################################################################
    innerAxes = axes(cur[1])[end-k:end-1] # effectively a set of colons of length k, to be used for input
    innerSub = (Base.OneTo(x) for x in outputSizes[i][end-k:end-1]) # effectively a set of colons of length k, to be used for the subsampled output

    outerAxes = axes(cur[1])[1:(end-k-1)] # the same idea as innerAxes, but for the example indices
    # precompute the wavelets
    if stType == "decreasing" && i >= 2
      numChildλ = numChildren(keeper, layers, nScalesLayers)
      daughters = computeWavelets(size(cur[1])[end-1], layers.shears[i], nScales = numChildλ)
    else
      daughters = computeWavelets(size(cur[1])[end-1], layers.shears[i], nScales = totalScales[i])
    end
    # TODO: compute a plan_fft that is efficient
    # compute the final mother if we're on the last layer
    if i == layers.m
      #println("On i=$(i), with nScales=1")
      mother = computeWavelets(size(cur[1])[end-1], layers.shears[i+1], nScales = 1)
      concatStartLast = sum(q[1:end-1] .* resultingSize[1:end-1]) + 1
      #println("last layer starting at $(concatStartLast)")
      nScalesLastLayer = getQ(layers, n, totalScales; product = false)[end-1]
      #println("nScalesLastLayer = $(nScalesLastLayer)")
    end
    concatStart = reduce(+, q[1:i-1] .* resultingSize[1:i-1], init = 0) + 1
    #println("starting at $(concatStart)")
    listOfProcessResults = Array{Future,1}(undef, size(cur[1])[end] * length(cur)) # parallel processing tools; the results are spread over the previous 2 layers paths
    # iterate over the scales two layers back
    for (j, x) in enumerate(cur)
      # iterate over all paths in layer i
      for λ = 1:size(x)[end]
        # do the actual transformation; the last layer requires far less saved than the mid-layers
        if i < layers.m
          if stType == "decreasing"
            futureThing = @spawn transformMidLayer!(layers, concatOutput, x, nextData, outerAxes, innerAxes, innerSub, k, i, nScalesLayers, daughters, totalScales[i], nonlinear, subsam, outputSubsample, outputSizes[i][1:end-1], λ, concatStart, keeper, j)
          else
            futureThing = @spawn transformMidLayer!(layers, concatOutput, x, dataSizes, outerAxes, innerAxes, innerSub, k, i, daughters, totalScales[i], nonlinear, subsam, outputSubsample, outputSizes[i][1:end-1], concatStart, λ, resultingSize, dataChannel)
          end
        else
          #TODO
          if stType == "decreasing"
            futureThing = @spawn transformFinalLayer!(layers, concatOutput, x, outerAxes, innerAxes, innerSub, k, i, daughters, totalScales[i], nonlinear, subsam, outputSubsample, λ, resultingSize, countInFinalLayer, concatStartLast, keeper, nScalesLastLayer)
          else
            futureThing = @spawn transformFinalLayer!(layers, concatOutput, x, outerAxes, innerAxes, innerSub, k, i, daughters, totalScales[i], nonlinear, subsam, outputSubsample, λ, resultingSize, concatStart, concatStartLast, nScalesLastLayer)
          end
        end
        if typeof(futureThing) <: typeof(Distributed.RemoteException)
          throw(futureThing)
        else
          listOfProcessResults[(j-1)*size(cur[1])[end]+λ] = futureThing
        end
      end
    end

    # using one channel per
    for (λ, x) in enumerate(listOfProcessResults)
      tmpFetch = fetch(x)
      if i < layers.m
        nextData[λ] = tmpFetch
      end
    end
  end
  return concatOutput
end




function transformMidLayer!(layers, concatOutput, curPath, dataSize, outerAxes, innerAxes, innerSub, k, i, daughters, nScales, nonlinear, subsam, outputSubsample, outputSizeThisLayer, concatStart, λ, resultingSize, homeChannel)
  # make an array to return the results
  toBeHandedBack = zeros(outputSizeThisLayer..., size(daughters, 2) - 1)
  # first perform the continuous wavelet transform on the data from the previous layer
  output = cwt(curPath[outerAxes..., innerAxes..., λ], layers.shears[i], daughters, nScales = nScales)
  innerMostAxes = axes(output)[end:end]
  # iterate over the non transformed dimensions of output
  for outer in eachindex(view(output, outerAxes..., [1 for i = 1:(k+1)]...))
    # output may have length zero if the starting path has low enough scale
    for j = 2:size(output)[end]
      toBeHandedBack[outer, innerSub..., j-1] = nonlinear.(subsam(output[outer, innerAxes..., j], layers.subsampling[i]))
    end
    # actually write to the output
    tmpOut = Array{Float64,k}(real(subsam(output[outer, innerAxes..., end], layers.subsampling[i])))
    sizeTmpOut = prod(size(tmpOut))
    if outputSubsample[1] > 1 || outputSubsample[2] > 1
      concatOutput[outer, concatStart.+(λ-1)*resultingSize[i].+(0:(resultingSize[i]-1))] = subsam(tmpOut, resultingSize[i], absolute = true)
    else
      concatOutput[outer, concatStart.+(λ-1)*resultingSize[i].+(0:sizeTmpOut-1)] = reshape(tmpOut, (sizeTmpOut))
    end
  end
  return toBeHandedBack
end




function transformFinalLayer!(layers, concatOutput, curPath, outerAxes, innerAxes, innerSub, k, i, daughters, nScales, nonlinear, subsam, outputSubsample, λ, resultingSize, concatStart, concatStartLast, nScalesLastLayer)
  localIndex = 0
  # iterate over the outerAxes
  # first perform the continuous wavelet transform on the data from the previous layer
  output = cwt(curPath[outerAxes..., innerAxes..., λ], layers.shears[i], daughters, nScales = nScales)
  for outer in eachindex(view(output, outerAxes..., [1 for i = 1:(k+1)]...))
    # write the output from the m-1st layer to output
    tmpOut = Array{Float64,k}(real(subsam(output[outer, innerAxes..., end], layers.subsampling[i])))
    if outputSubsample[1] > 1 || outputSubsample[2] > 1
      #println("outer, outPos[outer] = $((outer, outPos[outer]))")
      concatOutput[outer, concatStart.+(λ-1)*resultingSize[i].+(0:(resultingSize[i]-1))] = subsam(tmpOut, resultingSize[i], absolute = true)
    else
      sizeTmpOut = prod(size(tmpOut))
      concatOutput[outer, concatStart.+(λ-1)*resultingSize[i].+(0:(sizeTmpOut-1))] = reshape(tmpOut, (sizeTmpOut))
    end
  end
  # write the output from the mth layer to output
  for j = 2:size(output)[end]
    finalOutput = cwt(curPath[outerAxes..., innerAxes..., λ], layers.shears[i+1], daughters, nScales = 1)
    for outer in eachindex(view(finalOutput, outerAxes..., [1 for i = 1:(k+1)]...))
      tmpFinal = Array{Float64,k}(real(subsam(finalOutput[outer, innerAxes..., end], layers.subsampling[i])))
      if outputSubsample[1] > 1 || outputSubsample[2] > 1

        concatOutput[outer, concatStartLast+(λ-1)*nScalesLastLayer*resultingSize[i]+(j-2)*resultingSize[i].+(0:(resultingSize[i]-1))] = subsam(tmpFinal, resultingSize[i+1], absolute = true)
      else
        sizeTmpOut = prod(size(tmpFinal))
        concatOutput[outer, (concatStartLast+(λ-1)*nScalesLastLayer*resultingSize[i]+localIndex+(λ-1)*resultingSize[i+1]).+(0:(sizeTmpOut-1))] = reshape(tmpFinal, (sizeTmpOut))
      end
    end
  end
end
