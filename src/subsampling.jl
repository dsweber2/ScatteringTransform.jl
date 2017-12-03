# TODO: implement functions that return the size of the output these create
# Various methods of subsampling:
# maxPooling: chooses the maximum in a neighborhood large enough to shrink each dimension by a factor rate. Any dimension. Currently broken
# biliniear: resample on a grid with the specified rate, using a linear interpolation in both x and y between the 4 points. 2D only
# bspline: uses a quadratic bspline interpolation from the julia package Interpolations
# linear: resample on a grid with the specified rate using linear interpolation. 1D only
using Interpolations
import Base.maximum
# Define the maximum of a complex number to be the complex number with largest magnitude
maximum(A::Matrix{Complex{Float64}}) = A[indmax(abs(A))]
maximum(A::Array{Complex{Float64}}) = A[indmax(abs(A))]

# This one naively implements the sort of max pooling used in the CNN literature, and seems to have aliasing issues
"""
    output = maxPooling(input::Array{Complex128,2}, rate::Float64)
"""
function maxPooling(input::Array{Complex128,2}, rate::Float64)
  rows = Int32(ceil(size(input,1)./2.0))
  cols = Int32(ceil(size(input,2)./2.0))
  output = zeros(Complex128,rows,cols)
  for i=1:rows-1, j=1:cols-1
    rowcoord = Int64(floor(rate*(i-1)))+1:Int64(floor(rate*i))
    colcoord = Int64(floor(rate*(j-1)))+1:Int64(floor(rate*j))
    # println("$rowcoord,         $colcoord")
    output[i,j] = maximum(input[rowcoord,colcoord])
  end

  # A little bit extra on the bottom
  rowcoord = Int64(floor(rate*(rows-1)))+1:size(input,1)
  for j=1:cols-1
    colcoord = Int64(floor(rate*(j-1)))+1:Int64(floor(rate*j))
    output[end,j] = maximum(input[rowcoord,colcoord])
  end

  # A little bit extra on the right
  colcoord = Int64(floor(rate*(cols-1)))+1:size(input,2)
  for i=1:rows-1
    rowcord = Int64(floor(rate*(i-1)))+1:Int64(floor(rate*i))
    output[i,end] = maximum(input[rowcoord,colcoord])
  end

  # A little bit extra on the bottom right
  output[end,end] = maximum(input[rowcoord,colcoord])
  return output
end

"""
    output = bilinear(input::Array{Complex128,2}, rate::Float64)

  Use linear interpolation to evaluate the points off-grid for a2D scattering transform.
"""
function bilinear(input::Array{S,2}, rate::T) where {S<:Number,T<:Real}
  Nin = size(input)
  newSize = (Int64(ceil(size(input,1)/rate)),Int64(ceil(size(input,2)/rate)))
  evalPoints = Array{Tuple{Float64,Float64}}(newSize)
  output = zeros(Complex64,newSize)
  for (i,x)=enumerate(linspace(1,Nin[1],newSize[1])), (j,y)=enumerate(linspace(1,Nin[2],newSize[2]))
    evalPoints[i,j] = (x,y)
    xi = (modf(x)[1], Int64(modf(x)[2]))
    yi = (modf(y)[1], Int64(modf(y)[2]))
    if xi[2]<Nin[1] && yi[2]<Nin[2]
    output[i,j] = (1-xi[1])*(1-yi[1])*input[xi[2],yi[2]] + xi[1]*(1-yi[1])*input[xi[2]+1,yi[2]] + (1-xi[1])*yi[1]*input[xi[2],yi[2]+1] + xi[1]*yi[1]*input[xi[2]+1,yi[2]+1]
    elseif yi[2]<Nin[2]
      # just a linear interpolation vertically, as we're on an edge
      output[i,j] = (1-yi[1])*input[xi[2],yi[2]] + yi[1]*input[xi[2],yi[2]+1]
    elseif xi[2]<Nin[1]
      # just a linear interpolation horizontally, as we're on an edge
      output[i,j] = (1-xi[1])*input[xi[2],yi[2]] + xi[1]*input[xi[2]+1,yi[2]]
    else
      # we're just at the final gridpoint, stupid simple
      output[i,j] = input[xi[2],yi[2]]
    end
  end
  output
end

"""
  output = bspline(input::Array{S,1}, rate::T) where {S<:Number, T<:Real}

  subsample using a quadratic bspline interpolation, with reflection boundary conditions. rate must be ≧ 1.
"""
function bspline(input::Array{S,1}, rate::T) where {S<:Number, T<:Real}
  @assert rate>=1
  itp = interpolate(input,BSpline(Quadratic(Reflect())), OnGrid())
  itp[inspace(1,length(input),floor(length(input)./64))]
end
using Interpolations
t=-1:.001:1
x=1-t.^2+t.^3
itp = interpolate(x, BSpline(Quadratic(Reflect())), OnGrid())
plot(linspace(1,length(x),floor(length(x)./64)),itp[linspace(1,length(x),floor(length(x)./64))])
plot!(x)
using Plots
