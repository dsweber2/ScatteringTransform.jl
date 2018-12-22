module ScatteringTransform
using Interpolations, Wavelets, Distributed, JLD#, MAT, Plots, LaTeXStrings

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

# include("plotting.jl") # the current version of this only works with shearlets
export st, thinSt, flatten, MatrixAggrigator, plotShattered, plotCoordinate, reshapeFlattened, numberSkipped, logabs, maxPooling, ReLU, numScales, incrementKeeper, numInLayer

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
function st(X::Array{T}, layers::layeredTransform; nonlinear::Function=abs, subsam::Function=bspline, stType::String="full") where {T<:Real}
  numChildλ= 0
  # Insist that X has to have at least one extra meta-dimension, even if it is 1
  if size(X) == length(size(layers.subsampling))
    X = reshape(X, (1,size(X)...));
  end

  results = scattered(layers, X, stType)


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
      daughters = computeWavelets(cur[[1 for i=1:length(outerAxes)]..., innerAxes..., 1], layers.shears[i])
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
"""
function thinSt(X::Array{T}, layers::layeredTransform; nonlinear::Function=abs, subsam::Function=bspline, stType::String="full", outputSubsample::Tuple{Int,Int}=(-1,1)) where {T<:Real}
  # Insist that X has to have at least one extra meta-dimension, even if it is 1
  if size(X) == length(size(layers.subsampling))
    X = reshape(X, (1,size(X)...));
  end
  numChildλ= 0
  println("initial")
  #results = scattered(layers, X, stType)
  k = length(size(layers.subsampling))
  n = sizes(bspline, layers.subsampling, size(X)[(end-k+1):end])
  q = [numScales(layers.shears[i],n[i]) for i=1:layers.m+1]
  q = [prod(q[1:i-1].-1) for i=1:layers.m+1]
  dataSizes = [[size(X)[1:end-k]..., n[i], q[i]] for i=1:layers.m+1]
  outputSizes = [[size(X)[1:(end-k)]..., n[i+1], q[i]] for i=1:layers.m+1]
  println("the required sizes are dataSizes = $(dataSizes)")
  println("outputSizes = $(outputSizes)")
  nextData = reshape(X, (size(X)..., 1))
  # create list of size references if we're subsampling the output an extra amount
  if outputSubsample[1] > 1
    resultingSize = zeros(k)
    resultingSubsampling = zeros(k)
    # subsampling limited by the second entry
    for (i,x) in enumerate(outputSizes)
      proposedLevel = floor(Int, x[end-1])
      if proposedLevel < outputSubsample[2]
        # the result is smaller than the floor
        proposedLevel = outputSubsample[2]
      end
      resultingSize[i] = outputSubsample
    end
    println("resultingSize = $(resultingSize)")
  elseif outputSubsample[2] > 1
    resultingSize = outputSubsample[2]*ones(Int64,layers.m+1,k)
    println("resultingSize = $(resultingSize)")
  end
  if outputSubsample[1] > 1 || outputSubsample[2] > 1
    concatOutput = zeros(Float64, outputSizes[1][1:end-k-1]..., sum(resultingSize.*q))
  else
    concatOutput = zeros(Float64, outputSizes[1][1:end-k-1]..., sum([prod(outputSizes[end-k:end]) for x in outputSizes])...)
  end
  # keep track of where we are in each meta dimension
  outPos = ones(Int64, outputSizes[1][1:end-k-1]...)
  println("outPos = $(size(outPos))")
  println("outputSizes[1] = $(outputSizes[1])")
  println("size of concatOutput = size(concatOutput)")
  nScalesLayers = [numScales(layers.shears[i], max(dataSizes[i][1]-1,1)) for i=1:length(layers.shears)]
  for (i,layer) in enumerate(layers.shears)
    println("starting the actual transformation in layer $(i)")
    ######################################################################################################################################################################################################################################################################################
    cur = nextData #data from the previous layer
    if i <= layers.m
      nextData = zeros(dataSizes[i+1]...)
    end
    #println("size of nextData = $(size(nextData))")
    #println("size of cur $(size(cur))")
    # if we're decreasing, we need to keep track of what path the current index corresponds to
    if stType=="decreasing" && i>=2
      keeper = [1 for k=1:i-1]
    elseif stType=="decreasing" && i==1
      keeper = [1 for k=1:i]
    end

    innerAxes = axes(cur)[end-k:end-1] # effectively a set of colons of length k, to be used for input
        ######################################################################################################################################################################################################################################################################################
    innerSub = (Base.OneTo(x) for x in outputSizes[i][end-k:end-1])#axes(outputSizes[i]...)[end-k:end-1] # effectively a set of colons of length k, to be used for the subsampled output

    outerAxes = axes(cur)[1:(end-k-1)] # the same idea as innerAxes, but for the example indices
    #println(outerAxes)
    # precompute the wavelets
    if stType=="decreasing" && i>=2
      numChildλ = numChildren(keeper, layers, nScalesLayers)
      daughters = computeWavelets(cur[[1 for i=1:length(outerAxes)]..., innerAxes..., 1], layers.shears[i], J1=numChildλ - 1 + layers.shears[i].averagingLength)
    else
      daughters = computeWavelets(cur[[1 for i=1:length(outerAxes)]..., innerAxes..., 1], layers.shears[i])
    end
    #println("computed daughters in layer $(i)")
    # iterate over the outerAxes
    for outer in eachindex(view(cur, outerAxes..., [1 for i=1:k]..., 1))
      if i<=layers.m
        decreasingIndex = 1
        for λ = 1:size(cur)[end]
          # first perform the continuous wavelet transform on the data from the previous layer
          if stType=="decreasing" && i>=2
            numChildλ = numChildren(keeper, layers, nScalesLayers)
            #println("sizeCur = $(size(cur[outer, innerAxes..., λ]))")
            output=cwt(cur[outer, innerAxes...,λ], layers.shears[i], daughters, J1=numChildλ-1 +layers.shears[i].averagingLength)
            if size(output,2) != numChildλ+1
              error("size(output,2)!=numChildλ $(size(output,2))!=$(numChildλ+1)")
            end
          elseif stType=="decreasing"
            output = cwt(cur[outer, innerAxes..., λ], layers.shears[i], daughters)
          else
            output = cwt(cur[outer, innerAxes..., λ], layers.shears[i], daughters)
          end
          # output may have length zero if the starting path has low enough scale
          for j = 2:size(output,2)
            if stType == "full"
              #println("OUTER = $(outer)")
              #println(innerSub)
              #println((λ-1)*(size(output,2)-1)+j-1)
              #println("size of output $(size(output))")
              #println("size of output after subsampling $(size(subsam(output[innerAxes..., j], layers.subsampling[i])))")
              nextData[outer, innerSub..., (λ-1)*(size(output,2)-1)+j-1] = nonlinear.(subsam(output[innerAxes..., j], layers.subsampling[i]))
            else
              if stType=="decreasing"
                nextData[outer, innerSub..., decreasingIndex] = nonlinear.(subsam(output[innerAxes..., j], layers.subsampling[i]))
                decreasingIndex += 1
              end
            end
          end
          if stType=="decreasing" && i>1
            isEnd, keeper = incrementKeeper(keeper, i-1, layers, nScalesLayers)
          end
          tmpOut = Array{Float64,k}(real(subsam(output[innerAxes..., end], layers.subsampling[i])))
          sizeTmpOut = prod(size(tmpOut))
          if outputSubsample[1] > 1 || outputSubsample[2] > 1
            #println("outer, outPos[outer] = $((outer, outPos[outer]))")
              concatOutput[outer, outPos[outer].+(0:(resultingSize[i]-1))] = subsam(tmpOut, resultingSize[i], absolute = true)
            sizeTmpOut = resultingSize[i]
          else
            concatOutput[outer, outPos[outer].+(0:(sizeTmpOut-1))] = reshape(tmpOut, (sizeTmpOut))
          end
          outPos[outer] += sizeTmpOut
        end
      else
        for λ = 1:size(cur)[end]
          output = cwt(cur[outer, innerAxes..., λ], layers.shears[i], daughters, J1=1)
          tmpOut = Array{Float64,k}(real(subsam(output[innerAxes..., end], layers.subsampling[i])))
          sizeTmpOut = prod(size(tmpOut))
          if outputSubsample[1] > 1 || outputSubsample[2] > 1
            concatOutput[outer, outPos[outer].+(0:(resultingSize[i]-1))] = subsam(tmpOut, resultingSize[i], absolute = true)
            sizeTmpOut = resultingSize[i]
          else
            concatOutput[outer, outPos[outer].+(0:(sizeTmpOut-1))] = reshape(tmpOut, (sizeTmpOut))
          end
          outPos[outer] += sizeTmpOut
        end
      end
    end
  end
  return concatOutput
end



include("postProcessing.jl")
export logabs, ReLU, MatrixAggrigator, reshapeFlattened, loadSyntheticMatFile, transformFolder
end
