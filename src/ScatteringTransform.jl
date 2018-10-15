module ScatteringTransform
using Interpolations, Wavelets, Distributed #, JLD, MAT, Plots, LaTeXStrings

include("subsampling.jl")
export maxPooling, bilinear, bspline
include("modifiedTransforms.jl")
export CFWA, wavelet, cwt, getScales
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
# TODO: currently computing data at one layer deeper than the actual transform; there's two ways to fix that.
# order shouldn't matter since the types are unique
"""
  output = st(X::Array{Float64,1}, layers::layeredTransform; nonlinear::Function=abs, subsam::Function=subsample, stType::String, collator::Array{Array{float64},1}=Array{Array{float64},1}(0))

  1D scattering transform using the layeredTransform layers. you can switch out the nonlinearity as well as the method of subsampling. Finally, the stType is a string. If it is "full", it will produce all paths. If it is "decreasing", it will only keep paths of increasing scale. If it is "collating", you must also include a vector of matrices.
"""
function st(X::Array{T}, layers::layeredTransform; nonlinear::Function=abs, subsam::Function=bspline, stType::String="full", collator::Array{Array{Float64,2},1} = Array{Array{Float64,2},1}(undef, 0)) where {T<:Real}
  numChildλ= 0
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

    if i<=layers.m
      decreasingIndex = 1
      for λ = 1:size(cur,2)
        # first perform the continuous wavelet transform on the data from the previous layer
        if stType=="decreasing" && i>=2
          numChildλ = numChildren(keeper, layers, nScalesLayers)
          output = cwt(cur[:,λ], layers.shears[i], J1=numChildλ - 1 + layers.shears[i].averagingLength)
          if size(output,2) != numChildλ+1
            error("size(output,2)!=numChildλ $(size(output,2))!=$(numChildλ+1)")
          end
        elseif stType=="decreasing"
          output=cwt(cur[:,λ], layers.shears[i])
        else
          output = cwt(cur[:,λ], layers.shears[i])
        end
        # output may have length zero if the starting path has low enough scale
        for j = 2:size(output,2)
            if stType == "full"
              results.data[i+1][:,(λ-1)*(size(output,2)-1)+j-1] = nonlinear.(subsam(output[:,j], layers.subsampling[i]))
            else
              if stType=="decreasing"
                # println(decreasingIndex)
                results.data[i+1][:,decreasingIndex] = nonlinear.(subsam(output[:,j], layers.subsampling[i]))
                decreasingIndex += 1
              end
            end
        end
          if stType=="decreasing" && i>1
            println(keeper)
            isEnd, keeper = incrementKeeper(keeper, i-1, layers, nScalesLayers)
          end
        results.output[i][:,λ] = Array{Float64,1}(real(subsam(output[:,end], layers.subsampling[i])))
      end
    else
      # TODO: This is not an efficient implementation to get the last layer of output. There are several places where m+1 is substituted for m in the definition of shears to accomodate it. Data is shrunk by a layer in the sheared array
      for λ = 1:size(cur,2)
        output = cwt(cur[:,λ], layers.shears[i], J1=1)
        results.output[i][:,λ] = Array{Float64,1}(real(subsam(output[:,end], layers.subsampling[i])))
      end
    end
  end
  return results
end



st(layers::layeredTransform, X::Array{T,1}; nonlinear::Function=abs, subsam::Function=bspline) where {T<:Real} = st(X,layers,nonlinear=nonlinear, subsamp=subsam)


# TODO: make a function which determines the breakpoints given the layers function and the size of the input
"""
    thinSt(X::Array{T,1}, layers::layeredTransform; nonlinear::Function=abs, subsam::Function=bspline, stType::String="full", collator::Array{Array{Float64,2},1} = Array{Array{Float64,2},1}(0)) where {T<:Real}

  See the main st for a description of the options. This is a version of the 1D st that only returns a concatinated vector for output. It is most useful for classification applications.

"""
function thinSt(X::Array{T,1}, layers::layeredTransform; nonlinear::Function=abs, subsam::Function=bspline, stType::String="full", collator::Array{Array{Float64,2},1} = Array{Array{Float64,2},1}(0)) where {T<:Real}
  numChildλ= 0
  results = scattered(layers, X, stType)
  concatOutput = Vector{Float64}(sum([prod(size(x)) for x in results.output]))
  outPos = 1
  nScalesLayers = [numScales(layers.shears[i], size(results.data[i],1))-1 for i=1:length(layers.shears)]

  for (i,layer) in enumerate(layers.shears)
    cur = results.data[i] #data from the previous layer
    # if we're decreasing, we need to keep track of what path the current index corresponds to
    if stType=="decreasing" && i>=2
      keeper = [1 for k=1:i-1]
    elseif stType=="decreasing" && i==1
      keeper = [1 for k=1:i]
    end
    if i<=layers.m
      decreasingIndex = 1
      for λ = 1:size(cur,2)
        # first perform the continuous wavelet transform on the data from the previous layer
        if stType=="decreasing" && i>=2
          numChildλ = numChildren(keeper, layers, nScalesLayers)
          output=cwt(cur[:,λ], layers.shears[i], J1=numChildλ-1 +layers.shears[i].averagingLength)
          # println("numChildλ = $(numChildλ), keeper= $(keeper), size(output,2)-1 = $(size(output,2)-1)")
          if size(output,2) != numChildλ+1
            error("size(output,2)!=numChildλ $(size(output,2))!=$(numChildλ+1)")
          end
        elseif stType=="decreasing"
          output=cwt(cur[:,λ], layers.shears[i])
        else
          output = cwt(cur[:,λ], layers.shears[i])
        end
        # output may have length zero if the starting path has low enough scale
        for j = 2:size(output,2)
          if stType == "full"
            results.data[i+1][:,(λ-1)*(size(output,2)-1)+j-1] = nonlinear.(subsam(output[:,j], layers.subsampling[i]))
          else
            if stType=="decreasing"
              # println(decreasingIndex)
              results.data[i+1][:,decreasingIndex] = nonlinear.(subsam(output[:,j], layers.subsampling[i]))
              decreasingIndex += 1
            end
          end
        end
        results.output[i][:,λ] = Array{Float64,1}(real(subsam(output[:,end], layers.subsampling[i])))
        if stType=="decreasing" && i>1
          isEnd, keeper = incrementKeeper(keeper, i-1, layers, nScalesLayers)
        end
        tmpOut = Array{Float64,1}(real(subsam(output[:,end], layers.subsampling[i])))
        sizeTmpOut = prod(size(tmpOut))
        concatOutput[outPos+(0:sizeTmpOut-1)] = reshape(tmpOut, (prod(size(tmpOut))))
        outPos += sizeTmpOut
      end
    else
      # TODO: This is not an efficient implementation to get the last layer of output. There are several places where m+1 is substituted for m in the definition of shears to accomodate it. Data is shrunk by a layer in the sheared array
      for λ = 1:size(cur,2)
        output = cwt(cur[:,λ], layers.shears[i])
        tmpOut = Array{Float64,1}(real(subsam(output[:,end], layers.subsampling[i])))
        sizeTmpOut = prod(size(tmpOut))
        concatOutput[outPos+(0:sizeTmpOut-1)] = reshape(tmpOut, (sizeTmpOut))
        outPos += sizeTmpOut
      end
    end
  end
    return concatOutput
end



include("postProcessing.jl")
export logabs, ReLU, MatrixAggrigator, reshapeFlattened, loadSyntheticMatFile, transformFolder
end
