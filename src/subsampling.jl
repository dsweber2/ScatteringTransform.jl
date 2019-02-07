# TODO: implement functions that return the size of the output these create
# Various methods of subsampling:
# maxPooling: chooses the maximum in a neighborhood large enough to shrink each dimension by a factor rate. Any dimension. Currently broken
# biliniear: resample on a grid with the specified rate, using a linear interpolation in both x and y between the 4 points. 2D only
# bspline: uses a quadratic bspline interpolation from the julia package Interpolations
# linear: resample on a grid with the specified rate using linear interpolation. 1D only
import Base.maximum
# Define the maximum of a complex number to be the complex number with largest magnitude
maximum(A::Matrix{Complex{Float64}}) = A[indmax(abs(A))]
maximum(A::Array{Complex{Float64}}) = A[indmax(abs(A))]

# # This one naively implements the sort of max pooling used in the CNN literature, and seems to have aliasing issues
"""
    output = maxPooling(input::Array{T,2}, rate::Float64) where T<: Number
"""
function maxPooling(input::Array{T,2}, rate::Float64) where T<: Number
  rows = Int32(ceil(size(input,1) ./2.0))
  cols = Int32(ceil(size(input,2) ./2.0))
  output = zeros(T,rows,cols)
  for i=1:rows-1, j=1:cols-1
    rowcoord = Int(floor(rate*(i-1)))+1:Int(floor(rate*i))
    colcoord = Int(floor(rate*(j-1)))+1:Int(floor(rate*j))
    # println("$rowcoord,         $colcoord")
    output[i,j] = maximum(input[rowcoord,colcoord])
  end

  # A little bit extra on the bottom
  rowcoord = Int(floor(rate*(rows-1)))+1:size(input,1)
  for j=1:cols-1
    colcoord = Int(floor(rate*(j-1)))+1:Int(floor(rate*j))
    output[end,j] = maximum(input[rowcoord,colcoord])
  end

  # A little bit extra on the right
  colcoord = Int(floor(rate*(cols-1)))+1:size(input,2)
  for i=1:rows-1
    rowcord = Int(floor(rate*(i-1)))+1:Int(floor(rate*i))
    output[i,end] = maximum(input[rowcoord,colcoord])
  end

  # A little bit extra on the bottom right
  output[end,end] = maximum(input[rowcoord,colcoord])
  return output
end

"""
    output = bilinear(input::Array{S,2}, rate::T) where {S<:Number,T<:Real}

  Use linear interpolation to evaluate the points off-grid for a2D scattering transform.
"""
function bilinear(input::Array{S,2}, rate::T) where {S<:Number,T<:Real}
  Nin = size(input)
  newSize = (Int(ceil(size(input,1)/rate)),Int(ceil(size(input,2)/rate)))
  evalPoints = Array{Tuple{Float64,Float64}}(newSize)
  output = zeros(Complex64,newSize)
  for (i,x)=enumerate(linspace(1,Nin[1],newSize[1])), (j,y)=enumerate(linspace(1,Nin[2],newSize[2]))
    evalPoints[i,j] = (x,y)
    xi = (modf(x)[1], Int(modf(x)[2]))
    yi = (modf(y)[1], Int(modf(y)[2]))
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

@doc """
  output = bspline(input::Array{S,1}, rate::T) where {S<:Number, T<:Real}

  subsample using a quadratic bspline interpolation, with reflection boundary conditions. rate must be ≧ 1. If absolute is true, treat rate as just the total number of desired samples instead.
"""
function bspline(input::Array{S,1}, rate::T; absolute::Bool=false) where {S<:Number, T<:Real}
  @assert rate>=1
  itp = interpolate(input,BSpline(Quadratic(Reflect(OnGrid()))))
  if absolute
    return itp(range(1, stop = size(input)[end], length = rate))
  else
    return itp(range(1, stop = size(input)[end], length = floor(Int,length(input) ./rate)))
  end
end
# TODO: figure out what's wrong with Quadratic(Reflect() and re-implement it

@doc """
  subsamp = sizes(func::Function, rate::Array{T,1}, sizeX::Tuple{Int}) where T<:Real

  given a subsampling rate, a function func, and an initial size tuple sizeX, return an array of sizes as used by the scattering(1,2)D constructors
"""
function sizes(func::Function, rate::Array{T}, sizeX) where T<:Real
  if typeof(sizeX)<:Tuple
    sizeX = sizeX[end]
  end
  if func == bspline
    subsamp = zeros(Int,length(rate)+1); subsamp[1] = sizeX[1]
    for (i,rat) in enumerate(rate)
      subsamp[i+1] = floor(Int, subsamp[i] ./rat)
    end
  elseif func == bilinear
    subsamp = [Int( foldl((x,y)->ceil(x/y), sizeX, rate[1:i])) for i=0:length(rate)-1]
  else
    error("Sorry, that hasn't been implemented yet")
  end
  subsamp
end
# TODO: make the 2D version of this
# sizes(bspline, [2 2 2], size(x))
