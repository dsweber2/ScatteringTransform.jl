module ScatteringTransform
using Interpolations, Wavelets, JLD, MAT, Plots, LaTeXStrings
# using Shearlab, Interpolations, Wavelets, JLD, MAT, Plots, LaTeXStrings
interpolate(randn(1532) + randn(1532)*im, BSpline(Quadratic(Reflect())), OnGrid())
# using Interpolations, MAT
include("src/subsampling.jl")
innput = randn(1532) + randn(1532)*im
bspline(innput,8)
export maxPooling, bilinear, bspline
include("src/modifiedTransforms.jl")
export CFWA, wavelet, cwt, getScales
include("src/basicTypes.jl")
# X = testfunction(1532, "HeaviSine"); bspline(X, 8)
export layeredTransform, scattered1D, scattered2D
logabs(x)=log.(abs.(x))
ReLU(x::Array{Float64}) = max(0,x)
ReLU(x::Float64) = max(0,x)
include("src/Utils.jl")

# include("src/plotting.jl") # the current version of this only works with shearlets
export st, thinSt, flatten, MatrixAggrigator, plotShattered, plotCoordinate, reshapeFlattened, numberSkipped, logabs, maxPooling, ReLU, numScales

# TODO make a way to access a scattered2D by the index of (depth, scale,shearingFactor)

########################################################
# Actual scattering function
########################################################
########################################################
# TODO: currently computing data at one layer deeper than the actual transform; there's two ways to fix that.
"""
  output = st(X::Array{Float64,1}, layers::layeredTransform; nonlinear::Function=abs, subsam::Function=subsample, stType::String, collator::Array{Array{float64},1}=Array{Array{float64},1}(0))

  1D scattering transform using the layeredTransform layers. you can switch out the nonlinearity as well as the method of subsampling. Finally, the stType is a string. If it is "full", it will produce all paths. If it is "decreasing", it will only keep paths of increasing scale. If it is "collating", you must also include a vector of matrices.
"""
function st(X::Array{T,1}, layers::layeredTransform; nonlinear::Function=abs, subsam::Function=bspline, stType::String="full", collator::Array{Array{Float64,2},1} = Array{Array{Float64,2},1}(0)) where {T<:Real}

  results = scattered1D(layers, X, stType)
  for (i,layer) in enumerate(layers.shears)
    cur = results.data[i] #data from the previous layer
    if i<=layers.m
      # println("i=$i")
      decreasingIndex = 1
      for λ = 1:size(cur,2)
        # first perform the continuous wavelet transform on the data from the previous layer
        if stType=="decreasing"
          output=cwt(cur[:,λ], layers.shears[i], J1=numScales(layers.shears[i], length(cur[:,λ])) -λ+1)'
          println("size of len-λ+1 $(length(cur[:,λ])) -λ+1)")
        else
          output = cwt(cur[:,λ], layers.shears[i])'
        end
        # output may have length zero if the starting path has low enough scale
        if length(output) > 0
          # subsample each example, then apply the non-linearity
          for j = 1:size(output,2)-1
            # try
              if stType == "full"
                results.data[i+1][:,(λ-1)*(size(output,2)-1)+j] = nonlinear.(subsam(output[:,j], layers.subsampling[i]))
              else
                println("Size of output dim 2 $(size(output,2))")
                println("The index $(decreasingIndex*(size(output,2)-1)+j)")
                results.data[i+1][:,decreasingIndex*(size(output,2)-1)+j] = nonlinear.(subsam(output[:,j], layers.subsampling[i]))
                decreasingIndex+=1
              end
            # catch
              # println((λ-1)*(size(output,2)-1)+j)
              # error("bork")
            # end0
          end
          results.output[i][:,λ] = Array{Float64,1}(real(subsam(output[:,end], layers.subsampling[i])))
        end
      end
    else
      # TODO: This is not an efficient implementation to get the last layer of output. There are several places where m+1 is substituted for m in the definition of shears to accomodate it. Data is shrunk by a layer in the sheared array
      # println("i too big i=$(i)")
      for λ = 1:size(cur,2)
        output = cwt(cur[:,λ], layers.shears[i])'
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
    results = scattered1D(layers, X, stType)
    concatOutput = Vector{Float64}(sum([prod(size(x)) for x in results.output]))
    outPos = 1
    for (i,layer) in enumerate(layers.shears)
      cur = results.data[i] #data from the previous layer
      if i<=layers.m
        # println("i=$i")
        for λ = 1:size(cur,2)
          # first perform the continuous wavelet transform on the data from the previous layer
          if stType=="decreasing"
            output = cwt(cur[:,λ], layers.shears[i], J1=numScales(layers.shears[i],length(cur[:,λ])) -λ+1)'
          else
            output = cwt(cur[:,λ], layers.shears[i])'
          end
          # subsample each example, then apply the non-linearity
          for j = 1:size(output,2)-1
            if subsam == bspline
              results.data[i+1][:,(λ-1)*(size(output,2)-1)+j] = nonlinear.(subsam(output[:,j], layers.subsampling[i]))
            else
              error("the function $subsam isn't defined as a subsampling method at the moment")
            end
          end
          tmpOut = Array{Float64,1}(real(subsam(output[:,end], layers.subsampling[i])))
          sizeTmpOut = prod(size(tmpOut))
          concatOutput[outPos+(0:sizeTmpOut-1)] = reshape(tmpOut, (prod(size(tmpOut))))
          outPos += sizeTmpOut
          # results.output[i][:,λ] = Array{Float64,1}(real(subsam(output[:,end], layers.subsampling[i])))
        end
      else
        # TODO: This is not an efficient implementation to get the last layer of output. There are several places where m+1 is substituted for m in the definition of shears to accomodate it. Data is shrunk by a layer in the sheared array
        # println("i too big i=$(i)")
        for λ = 1:size(cur,2)
          output = cwt(cur[:,λ], layers.shears[i])'
          tmpOut = Array{Float64,1}(real(subsam(output[:,end], layers.subsampling[i])))
          sizeTmpOut = prod(size(tmpOut))
          concatOutput[outPos+(0:sizeTmpOut-1)] = reshape(tmpOut, (sizeTmpOut))
          outPos += sizeTmpOut
        end
      end
    end
    return concatOutput
end
include("src/postProcessing.jl")
export logabs, ReLU, MatrixAggrigator, reshapeFlattened, loadSyntheticMatFile, transformFolder
end
