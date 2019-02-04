module ScatteringTransform
using Distributed, SharedArrays
using Interpolations, Wavelets, Distributed, JLD, Plots#, MAT, Plots, LaTeXStrings



include("subsampling.jl")
export maxPooling, bilinear, bspline
include("modifiedTransforms.jl")
export CFWA, wavelet, cwt, getScales, computeWavelets
include("basicTypes.jl")
export layeredTransform, scattered, scattered2D
logabs(x)=log.(abs.(x))
ReLU(x::Float64) = max(0,x)
ReLU(x::A) where A<:Complex = real(x)>0 ? x : 0
include("Utils.jl")
export calculateThinStSizes
include("pathMethods.jl")
export pathType, pathToThinIndex

include("plotting.jl")
export st, thinSt, flatten, MatrixAggrigator, plotShattered, plotCoordinate, reshapeFlattened, numberSkipped, logabs, maxPooling, ReLU, numScales, incrementKeeper, numInLayer,  transformMidLayer!, transformFinalLayer!

# TODO make a way to access a scattered2D by the index of (depth, scale,shearingFactor)

########################################################
# Actual scattering function
########################################################
########################################################
# order shouldn't matter since the types are unique
"""
  output = st(X::Array{Float64,1}, layers::layeredTransform; nonlinear::Function=abs, subsam::Function=subsample, stType::String, collator::Array{Array{float64},1}=Array{Array{float64},1}(0))

  1D scattering transform using the layeredTransform layers. you can switch out the nonlinearity as well as the method of subsampling. Finally, the stType is a string. If it is "full", it will produce all paths. If it is "decreasing", it will only keep paths of increasing scale. If it is "collating", you must also include a vector of matrices.
"""
function st(X::Array{T}, layers::layeredTransform; nonlinear::Function=abs, subsam::Function=bspline, stType::String="full", totalScales = [NaN for i=1:layers.m]) where {T<:Real}
  numChildλ= 0
  # Insist that X has to have at least one extra meta-dimension, even if it is 1
  if size(X) == length(size(layers.subsampling))
    X = reshape(X, (1,size(X)...));
  end

  results = scattered(layers, X, stType)

  si
  nScalesLayers = [numScales(layers.shears[i], size(results.data[i],1))-1 for i=1:length(layers.shears)]
  for (i,layer) in enumerate(layers.shears)
    cur = results.data[i] #data from the previous layer
    # if we're decreasing, we need to keep track of what path the current index corresponds to
    if stType=="decreasing" && i>=2
      keeper = [1 for k=1:i-1]
    elseif i==1 && stType=="decreasing"
      keeper = [1 for k=1:i]
    end

    innerAxes = axes(cur)[end-results.k:end-1] # effectively a set of colons of length k, to be used for input
    innerSub = axes(results.output[i])[end-results.k:end-1] # effectively a set of colons of length k, to be used for the subsampled output

    outerAxes = axes(cur)[1:(end-results.k-1)] # the same idea as innerAxes, but for the example indices
    # precompute the wavelets
    if stType=="decreasing" && i>=2
      numChildλ = numChildren(keeper, layers, nScalesLayers)
      daughters = computeWavelets(cur[[1 for i=1:length(outerAxes)]..., innerAxes..., 1], layers.shears[i], J1=numChildλ - 1 + layers.shears[i].averagingLength)
    else
      daughters = computeWavelets(cur[[1 for i=1:length(outerAxes)]..., innerAxes..., 1], layers.shears[i],J1=totalScales[i])
    end

    # iterate over the outerAxes
    for outer in eachindex(view(cur, outerAxes..., [1 for i=1:results.k]..., 1))
      if i<=layers.m
        decreasingIndex = 1
        for λ = 1:size(cur)[end]
          # first perform the continuous wavelet transform on the data from the previous layer
          if stType=="decreasing" && i>=2
            numChildλ = numChildren(keeper, layers, nScalesLayers)
            output = cwt(cur[outer, innerAxes...,λ], layers.shears[i], daughters, J1=numChildλ - 1 + layers.shears[i].averagingLength)
            if size(output,2) != numChildλ+1
              error("size(output,2)!=numChildλ $(size(output,2))!=$(numChildλ+1)")
            end
          elseif stType=="decreasing"
            output=cwt(cur[outer, innerAxes..., λ], layers.shears[i], daughters)
          else
            output = cwt(cur[outer, innerAxes..., λ], layers.shears[i], daughters)
          end
          # output may have length zero if the starting path has low enough scale
          for j = 2:size(output,2)
            if stType == "full"
              results.data[i+1][outer, innerSub..., (λ-1)*(size(output,2)-1)+j-1] = nonlinear.(subsam(output[innerAxes..., j], layers.subsampling[i]))
            else
              if stType=="decreasing"
                results.data[i+1][outer, innerSub..., decreasingIndex] = nonlinear.(subsam(output[innerAxes..., j], layers.subsampling[i]))
                decreasingIndex += 1
              end
            end
          end
          if stType=="decreasing" && i>1
            isEnd, keeper = incrementKeeper(keeper, i-1, layers, nScalesLayers)
          end
          results.output[i][outer, innerSub..., λ] = Array{Float64,results.k}(real(subsam(output[innerAxes..., end], layers.subsampling[i])))
        end
      else
        for λ = 1:size(cur)[end]
          output = cwt(cur[outer, innerAxes..., λ], layers.shears[i], daughters, J1=1)
          results.output[i][outer, innerSub..., λ] = Array{Float64,results.k}(real(subsam(output[innerAxes..., end], layers.subsampling[i])))
        end
      end
    end
  end
  return results
end



st(layers::layeredTransform, X::Array{T}; nonlinear::Function=abs, subsam::Function=bspline) where {T<:Real} = st(X,layers,nonlinear=nonlinear, subsamp=subsam)




# TODO: make a function which determines the breakpoints given the layers function and the size of the input
"""
    thinSt(X::Array{T}, layers::layeredTransform; nonlinear::Function=abs, subsam::Function=bspline, stType::String="full", outputSubsample::Tuple{Int,Int}=(-1,-1)) where {T<:Real}

  See the main st for a description of the options. This is a version of the 1D st that only returns a concatinated vector for output. It is most useful for classification applications.

outputSubsample is a tuple, with the first number indicating a rate, and the second number giving the minimum allowed size. If some of the entries are less than 1, it has different behaviour:
(<1, x): subsample to x elements for each path.
(<1, <1): no ssubsampling
(x, <1) subsample at a rate of x, with at least one element kept in each path

totalScales, if positive, gives the number of non-averaging wavelets.
"""
function thinSt(X::Array{T}, layers::layeredTransform; nonlinear::Function=abs, subsam::Function=bspline, stType::String="full", outputSubsample::Tuple{Int,Int}=(-1,1), totalScales = [-1 for i=1:layers.m+1]) where {T<:Real}
  @assert length(totalScales)==layers.m+1
  # Insist that X has to have at least one extra meta-dimension, even if it is 1
  if size(X) == length(size(layers.subsampling))
    X = reshape(X, (1,size(X)...));
  end
  numChildλ = 0
  #println("initial")
  #results = scattered(layers, X, stType)
  #println("size(X) = $(size(X))")
  k, n, q, dataSizes, outputSizes, resultingSize = calculateThinStSizes(layers, outputSubsample, size(X), totalScales = totalScales)
  #println("the required sizes are dataSizes = $(dataSizes)")
  #println("outputSizes = $(outputSizes)")
  nextData = [reshape(X, (size(X)..., 1)),]
  # create list of size references if we're subsampling the output an extra amount

  if outputSubsample[1] > 1 || outputSubsample[2] > 1
    concatOutput = SharedArray(zeros(Float64, outputSizes[1][1:end-k-1]..., sum(resultingSize.*q)))
  else
    concatOutput = SharedArray(zeros(Float64, outputSizes[1][1:end-k-1]..., sum([prod(x[end-k:end]) for x in outputSizes])...))
  end
  # keep track of where we are in each meta dimension
  outPos = ones(Int64, outputSizes[1][1:end-k-1]...)
  nScalesLayers = [numScales(layers.shears[i], max(dataSizes[i][1]-1,1)) for i=1:length(layers.shears)]
  for (i,layer) in enumerate(layers.shears[1:layers.m])
    #println("starting the actual transformation in layer $(i)")
    cur = nextData #data from the previous layer
    #println("size(cur) = $(size(cur[1]))")
    # only store the intermediate results in intermediate layers
    if i < layers.m
      # store the data in a channel, along with the number of it's parent
      dataChannel = Channel{Tuple{Array{length(dataSizes[i+1])},Int64}}(size(cur)[end])
      nextData = Array{SharedArray{Float64, length(dataSizes[i+1])},1}(undef, size(cur[1])[end]*length(cur))
    end
    # if we're decreasing, we need to keep track of what path the current index corresponds to
    if stType=="decreasing" && i>=2
      keeper = [1 for k=1:i-1]
    elseif stType=="decreasing" && i==1
      keeper = [1 for k=1:i]
    end

    innerAxes = axes(cur[1])[end-k:end-1] # effectively a set of colons of length k, to be used for input
    ######################################################################################################################################################################################################################################################################################
    innerSub = (Base.OneTo(x) for x in outputSizes[i][end-k:end-1]) # effectively a set of colons of length k, to be used for the subsampled output
    
    outerAxes = axes(cur[1])[1:(end-k-1)] # the same idea as innerAxes, but for the example indices
    #println(outerAxes)
    # precompute the wavelets
    if stType=="decreasing" && i>=2
      numChildλ = numChildren(keeper, layers, nScalesLayers)
      daughters = computeWavelets(size(cur[1])[end-1], layers.shears[i], nScales=numChildλ)
    else
      daughters = computeWavelets(size(cur[1])[end-1], layers.shears[i],nScales=totalScales[i])
    end

    # compute the final mother if we're on the last layer
    if i==layers.m
      #println("On i=$(i), with nScales=1")
      mother = computeWavelets(size(cur[1])[end-1], layers.shears[i+1], nScales=1)
      concatStartLast = sum(q[1:end-1].*resultingSize[1:end-1])+1
      #println("last layer starting at $(concatStartLast)")
      nScalesLastLayer = getQ(layers, n, totalScales; product=false)[end-1]
      #println("nScalesLastLayer = $(nScalesLastLayer)")  
    end
    concatStart = sum(q[1:i-1].*resultingSize[1:i-1])+1
    #println("starting at $(concatStart)")
    listOfProcessResults = Array{Future, 1}(undef, size(cur[1])[end]*length(cur)) # parallel processing tools; the results are spread over the previous 2 layers paths
    # iterate over the scales two layers back
    for (j,x) in enumerate(cur)
      # iterate over all paths in layer i
      for λ = 1:size(x)[end]
        # do the actual transformation; the last layer requires far less saved than the mid-layers
        if i<layers.m
          if stType == "decreasing"
            listOfProcessResults[(j-1)*size(cur[1])[end] + λ] = @spawn transformMidLayer!(layers, concatOutput, x, nextData, outerAxes, innerAxes, innerSub, k, i, nScalesLayers, daughters, totalScales[i], nonlinear, subsam, outputSubsample, outputSizes[i][1:end-1], λ, concatStart, keeper, j)
          else
            listOfProcessResults[(j-1)*size(cur[1])[end] + λ] = @spawn transformMidLayer!(layers, concatOutput, x, dataSizes, outerAxes, innerAxes, innerSub, k, i, daughters, totalScales[i], nonlinear, subsam, outputSubsample, outputSizes[i][1:end-1], concatStart, λ, resultingSize, dataChannel)
          end
        else
          #TODO
          if stType=="decreasing"
            listOfProcessResults[(j-1)*size(cur[1])[end] + λ] = @spawn transformFinalLayer!(layers, concatOutput, x, outerAxes, innerAxes, innerSub, k, i, daughters, totalScales[i], nonlinear, subsam, outputSubsample, λ, resultingSize, countInFinalLayer, concatStartLast, keeper, nScalesLastLayer)
          else
            listOfProcessResults[(j-1)*size(cur[1])[end] + λ] = @spawn transformFinalLayer!(layers, concatOutput, x, outerAxes, innerAxes, innerSub, k, i, daughters, totalScales[i], nonlinear, subsam, outputSubsample, λ, resultingSize, concatStart, concatStartLast, nScalesLastLayer)
          end
        end
      end
    end
    
    # using one channel per
    for (λ,x) in enumerate(listOfProcessResults)
      tmpFetch = fetch(x)
      if i<layers.m
        nextData[λ] = tmpFetch
      end
    end
  end
return concatOutput
end

"""
    transformMidLayer!(layers, concatOutput, cur, dataSize, outerAxes, innerAxes, innerSub, k, i, daughters, nScales, nonlinear, subsam, outputSubsample, concatStart, λ)
    transformMidLayer!(layers, concatOutput, cur, nextData, outerAxes, innerAxes, innerSub, outerAxes, k, i, nScalesLayers, daughters, nScales, nonlinear, subsam, outputSubsample,
 λ, keeper)
An intermediate function which takes a single path in a layer and generates all of the children of that path in the next layer. Only to be used on intermediate layers, as finalLayerTransform fills the same role in the last layer. If keeper is omitted, it assumes we don't have decreasing type, otherwise it's assumed

TODO: figure out why we get negative values in the first layer. That's weird and shouldn't happen.
"""
function transformMidLayer!(layers, concatOutput, curPath, dataSize, outerAxes, innerAxes, innerSub, k, i, daughters, nScales, nonlinear, subsam, outputSubsample, outputSizeThisLayer, concatStart, λ, resultingSize, homeChannel)
  # make an array to return the results
  toBeHandedBack = zeros(outputSizeThisLayer..., size(daughters,2)-1)
  # iterate over the outerAxes
  for outer in eachindex(view(curPath, outerAxes..., [1 for i=1:k]..., 1))
    # first perform the continuous wavelet transform on the data from the previous layer
    output = cwt(curPath[outer, innerAxes..., λ], layers.shears[i], daughters, nScales=nScales)
    # output may have length zero if the starting path has low enough scale
    for j = 2:size(output)[end]
      toBeHandedBack[outer, innerSub..., j-1] = nonlinear.(subsam(output[innerAxes..., j], layers.subsampling[i]))
    end
    # actually write to the output
    tmpOut = Array{Float64,k}(real(subsam(output[innerAxes..., end], layers.subsampling[i])))
    sizeTmpOut = prod(size(tmpOut))
    if outputSubsample[1] > 1 || outputSubsample[2] > 1
      concatOutput[outer, concatStart .+ (λ-1)*resultingSize[i] .+ (0:(resultingSize[i]-1))] = subsam(tmpOut, resultingSize[i], absolute = true)
    else
      concatOutput[outer, concatStart .+ (λ-1)*resultingSize[i] .+ (0:sizeTmpOut-1)] = reshape(tmpOut, (sizeTmpOut))
    end
  end
  return toBeHandedBack
end

"""
    transformFinalLayer!(layers, concatOutput, curPath, outerAxes, innerAxes, innerSub, k, i, daughters, nScales, nonlinear, subsam, outputSubsample, λ, resultingSize, concatStart, λ)
Do the transformation of the m-1 to mth layer, the output of the m-1st layer, and the output of the mth layer. These are glued together for mostly for computational reasons. There is currently no decreasing paths version, so it should end up throwing errors
"""
function transformFinalLayer!(layers, concatOutput, curPath, outerAxes, innerAxes, innerSub, k, i, daughters, nScales, nonlinear, subsam, outputSubsample, λ, resultingSize, concatStart, concatStartLast, nScalesLastLayer)
  localIndex = 0
  # iterate over the outerAxes
  for outer in eachindex(view(curPath, outerAxes..., [1 for i=1:k]..., 1))
    # first perform the continuous wavelet transform on the data from the previous layer
    output = cwt(curPath[outer, innerAxes..., λ], layers.shears[i], daughters, nScales=nScales)
    # write the output from the m-1st layer to output
    tmpOut = Array{Float64,k}(real(subsam(output[innerAxes..., end], layers.subsampling[i])))
    if outputSubsample[1] > 1 || outputSubsample[2] > 1
      #println("outer, outPos[outer] = $((outer, outPos[outer]))")
      concatOutput[outer, concatStart .+ (λ-1)*resultingSize[i] .+ (0:(resultingSize[i]-1))] = subsam(tmpOut, resultingSize[i], absolute = true)
    else
      sizeTmpOut = prod(size(tmpOut))
      concatOutput[outer, concatStart .+ (λ-1)*resultingSize[i].+(0:(sizeTmpOut-1))] = reshape(tmpOut, (sizeTmpOut))
    end
    # write the output from the mth layer to output
    for j = 2:size(output)[end]
      finalOutput = cwt(curPath[outer, innerAxes..., λ], layers.shears[i+1], daughters, nScales=1)
      tmpFinal = Array{Float64,k}(real(subsam(finalOutput[innerAxes..., end], layers.subsampling[i])))
      if outputSubsample[1] > 1 || outputSubsample[2] > 1
        
        concatOutput[outer, concatStartLast + (λ-1)*nScalesLastLayer*resultingSize[i] + (j-2)*resultingSize[i] .+ (0:(resultingSize[i]-1))] = subsam(tmpFinal, resultingSize[i+1], absolute = true)
      else
        sizeTmpOut = prod(size(tmpFinal))
        concatOutput[outer, (concatStartLast + (λ-1)*nScalesLastLayer*resultingSize[i] + localIndex + (λ-1)*resultingSize[i+1]).+(0:(sizeTmpOut-1))] = reshape(tmpFinal, (sizeTmpOut))
        localIndex += sizeTmpOut
      end
    end
    localIndex = size(output)[end]
  end
end

# function transformMidLayer!(layers, concatOutput, cur, nextData, outerAxes, innerAxes, innerSub, k, i, nScalesLayers, daughters, J1, nonlinear, subsam, outputSubsample, λ, concatStart, keeper)
#    # TODO: This (the decreasing paths version) is utterly broken atm. Will return and fix it
#   decreasingIndex = 1
#   # iterate over the outerAxes
#   for outer in eachindex(view(cur, outerAxes..., [1 for i=1:k]..., 1))
#     # first perform the continuous wavelet transform on the data from the previous layer
#     if i>=2
#       output=cwt(cur[outer, innerAxes...,λ], layers.shears[i], daughters, J1=J1)
#     else
#       output = cwt(cur[outer, innerAxes..., λ], layers.shears[i], daughters)
#     end
#     # output may have length zero if the starting path has low enough scale
#     for j = 2:size(output,2)
#       nextData[outer, innerSub..., decreasingIndex] = nonlinear.(subsam(output[innerAxes..., j], layers.subsampling[i]))
#       decreasingIndex += 1
#     end
#     if i>1
#       isEnd, keeper = incrementKeeper(keeper, i-1, layers, nScalesLayers)
#     end
#     tmpOut = Array{Float64,k}(real(subsam(output[innerAxes..., end], layers.subsampling[i])))
#     sizeTmpOut = prod(size(tmpOut))
#     if outputSubsample[1] > 1 || outputSubsample[2] > 1
#       concatOutput[outer, concatStart .+ (0:(resultingSize[i]-1))] = subsam(tmpOut, resultingSize[i], absolute = true)
#       sizeTmpOut = resultingSize[i]
#     else
#       #TODO this is a sequential way of accessing the location
#       concatOutput[outer, concatStart.+(0:(sizeTmpOut-1))] = reshape(tmpOut, (sizeTmpOut))
#     end
#   end
# end



include("postProcessing.jl")
export logabs, ReLU, MatrixAggrigator, reshapeFlattened, loadSyntheticMatFile, transformFolder
end
