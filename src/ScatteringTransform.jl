module ScatteringTransform
using Interpolations, Wavelets, Distributed #, JLD, MAT, Plots, LaTeXStrings

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
    thinSt(X::Array{T}, layers::layeredTransform; nonlinear::Function=abs, subsam::Function=bspline, stType::String="full") where {T<:Real}

  See the main st for a description of the options. This is a version of the 1D st that only returns a concatinated vector for output. It is most useful for classification applications.

"""
function thinSt(X::Array{T}, layers::layeredTransform; nonlinear::Function=abs, subsam::Function=bspline, stType::String="full") where {T<:Real}
  # Insist that X has to have at least one extra meta-dimension, even if it is 1
  if size(X) == length(size(layers.subsampling))
    X = reshape(X, (1,size(X)...));
  end
  numChildλ= 0
  results = scattered(layers, X, stType)
  concatOutput = zeros(Float64, size(results.output[1])[1:end-results.k-1]..., sum([prod(size(x)[end-results.k:end]) for x in results.output])...)
  # keep track of where we are in each meta dimension
  outPos = ones(Int64, size(results.output[1])[1:end-results.k-1]...)
  nScalesLayers = [numScales(layers.shears[i], size(results.data[i],1))-1 for i=1:length(layers.shears)]
  for (i,layer) in enumerate(layers.shears)
    cur = results.data[i] #data from the previous layer
    # if we're decreasing, we need to keep track of what path the current index corresponds to
    if stType=="decreasing" && i>=2
      keeper = [1 for k=1:i-1]
    elseif stType=="decreasing" && i==1
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
          tmpOut = results.output[i][outer, innerSub..., λ]
          sizeTmpOut = prod(size(tmpOut))
          if outer==2
            println("outputPos = $(outPos)")
            println("tmpOut max should be = $(maximum(reshape(tmpOut, (sizeTmpOut))))")
            println("tmpOut max is = $(maximum(reshape(tmpOut, (prod(size(tmpOut))))))")
          end
          concatOutput[outer, outPos[outer].+(0:(sizeTmpOut-1))] = reshape(tmpOut, (sizeTmpOut))
          outPos[outer] += sizeTmpOut
        end
      else
        for λ = 1:size(cur)[end]
          output = cwt(cur[outer, innerAxes..., λ], layers.shears[i], daughters, J1=1)
          results.output[i][outer, innerSub..., λ] = Array{Float64,results.k}(real(subsam(output[innerAxes..., end], layers.subsampling[i])))
          tmpOut = results.output[i][outer, innerSub..., λ]
          sizeTmpOut = prod(size(tmpOut))
          if outer==2
            println("outputPos = $(outPos)")
            println("tmpOut max should be = $(maximum(reshape(tmpOut, (sizeTmpOut))))")
            println("tmpOut max is = $(maximum(reshape(tmpOut, (prod(size(tmpOut))))))")
          end
          concatOutput[outer, outPos[outer].+(0:sizeTmpOut-1)] = reshape(tmpOut, (sizeTmpOut))
          outPos[outer] += sizeTmpOut
        end
      end
    end
  end
  return concatOutput
  #return results
end
11508


include("postProcessing.jl")
export logabs, ReLU, MatrixAggrigator, reshapeFlattened, loadSyntheticMatFile, transformFolder
end
