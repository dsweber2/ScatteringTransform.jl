using Wavelets, Plots, DSP, Shearlab

include("src/modifiedTransforms.jl")
module ScatteringTransform
using Shearlab, JLD, MAT, Plots, LaTeXStrings, Interpolations, Wavelets
include("src/Utils.jl")
export logabs, ReLU, MatrixAggrigator, reshapeFlattened
include("src/modifiedTransforms.jl")
export CFWA, wavelet, cwt, getScales
include("src/basicTypes.jl")
include("plotting.jl")
include("subsampling.jl")
export layeredTransform, scattered1D, scattered2D, scattering, MatrixAggrigator, plotShattered, plotCoordinate, reshapeFlattened, numberSkipped, logabs, maxPooling, ReLU


# TODO make a way to access a shearedArray by the index of (depth, scale,shearingFactor)

########################################################
# Actual Shattering function
########################################################
########################################################
# TODO: currently computing data at one layer deeper than the actual transform; there's two ways to fix that.
"""
  output = shatter(X::Array{Float64,2}, layers::layeredTransform; nonlinear::Function=abs, subsam::Function=subsample)
"""
function st(X::Array{T,2}, layers::layeredTransform; nonlinear::Function=abs, subsam::Function=bilinear) where {T<:Real}
  results = shearedArray(layers, X)
  for (i,layer) in enumerate(layers.shears)
    cur = results.data[i] #data from the previous layer
    if i<size(layers.shears,1)
      # println("i=$i")
      for λ = 1:size(cur,3)
        # first perform the shearing transform on the data from the previous layer
        output = Shearlab.SLsheardec2D(cur[:,:,λ], layers.shears[i])
        # if using interpolations, prepare knots beforehand
        # if subsam==subsample
        #   knots = ([x for x=1:size(output[:,:,1],1)], [x for x=1:size(output[:,:,1],2)])
        # end
        # subsample each example, then apply the non-linearity
        for j = 1:size(output,3)-1
          if subsam== bilinear
            results.data[i+1][:,:,(λ-1)*(size(output,3)-1)+j] = nonlinear.(subsam(output[:,:,j],layers.subsampling[i]))
          else
            results.data[i+1][:,:,(λ-1)*(size(output,3)-1)+j] = nonlinear.(subsam(output[:,:,j],layers.subsampling[i]))
          end
        end
        results.output[i][:,:,λ] = Array{Float64,2}(real(subsam(output[:,:,end],layers.subsampling[i])))
      end
    else
      # TODO: This is not an efficient implementation to get the last layer of output. There are several places where m+1 is substituted for m in the definition of shears to accomodate it. Data is shrunk by a layer in the sheared array
      println("i too big i=$(i)")
      for λ = 1:size(cur,3)
        output = Shearlab.SLsheardec2D(cur[:,:,λ], layers.shears[i])
        results.output[i][:,:,λ] = Array{Float64,2}(real(subsam(output[:,:,end],layers.subsampling[i])))
      end
    end
  end
  return results
end
function st(layers::layeredTransform, X::Array{T,2}; nonlinear::Function=abs, subsam::Function=bilinear) where {T<:Real}
   return st(X, layers; nonlinear=nonlinear, subsam=subsam)
end




########################################################
# Utilities
########################################################
########################################################

# simply take the log of the absolute value

end
