"""

  given a path p and a flattened scattering transform flat, return the transform of that row.
"""
function findp(flat::Vector{T}, p::Vector{S}, layers::layeredTransform) where {T <: Number, S<:Integer}

end


"""
  p = computePath(layers::layeredTransform, layerM::Int, λ::Int; stType::String="full"))

compute the path p that corresponds to location λ in the stType of transform
"""
function computePath(layers::layeredTransform, layered1D::scattered{T,1}, layerM::Int, λ::Int; stType::String="full") where T<:Number
  nScalesLayers = [numScales(layers.shears[i], layered1D.data[i]) for i=1:layerM]
  p = zeros(layerM)
  if stType=="full"
    # start at the outermost level and work in
    for i=1:layerM
      p[i] = div(tmpλ, prod(nScalesLayers[i+1:end]))+1
      tmpλ -= (p[i]-1)*prod(nScalesLayers[i+1:end])
    end
  elseif stType=="decreasing"

  end
end


@doc """
    numChildren = numChildren(keeper::Array{Int}, layers::layeredTransform, nScalesLayers::Array{Int})

given a keeper, determine the number of children it has.
"""
function numChildren(keeperOriginal::Array{Int}, scalingFactors::Array{Float64}, nScalesLayers::Array{Int})
  keeperOriginal = [keeperOriginal; 1]
  keeper = copy(keeperOriginal)
  m = length(keeper)
  if m==0
    return 1 # not sure why you bothered
  elseif m==1
    numThisLayer = 1
  else
    numThisLayer = 0
  end
  while true
    if keeperOriginal==[12, 12, 1]
    end
    # we're done if either if we've hit the maximum number of scales, or if at least one of the old indices is larger than it used to be. In the first case, count it, in the second case, don't
    if keeper[m]==nScalesLayers[m]
      return numThisLayer+=1
    elseif reduce((a,b)->a|b, [keeper[i]>keeperOriginal[i] for i=1:m-1])
      if keeperOriginal==[12, 12, 1]
      end
      return numThisLayer
    end
    numThisLayer += 1
    isLast, keeper = incrementKeeper(keeper, m, scalingFactors, nScalesLayers)
    if isLast
      break
    end
  end
  numThisLayer
end
numChildren(keeperOriginal::Array{Int}, layers::layeredTransform, nScalesLayers::Array{Int}) = numChildren(keeperOriginal::Array{Int}, [shear.scalingFactor for shear in layers.shears], nScalesLayers::Array{Int})

@doc """
numInLayer(m::Int, layers::layeredTransform, nScalesLayers::Array{Int})

A stupidly straightforward way of counting the number of decreasing paths in a given layer.
"""
function numInLayer(m::Int, scalingFactors::Array{Float64}, nScalesLayers::Array{Int})
  keeper = [1 for i=1:m]
  if m==0
    return 1 # not sure why you bothered
  else
    numThisLayer = 1
  end
  while true
    isLast, keeper = incrementKeeper(keeper, m, scalingFactors, nScalesLayers)
      if isLast
      break
    end
    numThisLayer+=1
  end
  numThisLayer
end
numInLayer(m::Int, layers::layeredTransform, nScalesLayers::Array{Int}) = numInLayer(m, [shear.scalingFactor for shear in layers.shears], nScalesLayers)


@doc """
    k, n, q, dataSizes, outputSizes, resultingSize = calculateThinStSizes(layers, outputSubsample, Xsize; totalScales=[NaN for i=1:layers.m])
"""
function calculateThinStSizes(layers, outputSubsample, Xsize; totalScales=[-1 for i=1:layers.m+1])
  k = length(size(layers.subsampling))
  n = Int.(sizes(bspline, layers.subsampling, Xsize[(end-k+1):end]))
  q = getQ(layers,n,totalScales)
  dataSizes = [Int.([Xsize[1:end-k]..., n[i], q[i]]) for i=1:layers.m+1]
  outputSizes = [Int.([Xsize[1:(end-k)]..., n[i+1], q[i]]) for i=1:layers.m+1]
  if outputSubsample[1] > 1
    resultingSize = zeros(layers.m+1)
    resultingSubsampling = zeros(layers.m+1)
    # subsampling limited by the second entry
    for (i,x) in enumerate(outputSizes)
      proposedLevel = floor(Int, x[end-1]/outputSubsample[1])
      if proposedLevel < outputSubsample[2]
        # the result is smaller than the floor
        proposedLevel = outputSubsample[2]
      end
      resultingSize[i] = proposedLevel
    end
    # println("resultingSize = $(resultingSize)")
  elseif outputSubsample[2] > 1
    resultingSize = outputSubsample[2]*ones(Int64,layers.m+1,k)
    # println("resultingSize = $(resultingSize)")
  else
    resultingSize = outputSizes
  end
  return (k, n, q, dataSizes, outputSizes, resultingSize)
end

@doc """
  q = getQ(layers, n, totalScales; product=true)
calculate the total number of entries in each layer
"""
function getQ(layers, n, totalScales; product=true)
  q = [numScales(layers.shears[i],n[i]) for i=1:layers.m+1]
  q = [(isnan(totalScales[i]) || totalScales[i]<=0) ? q[i] : totalScales[i] for i=1:layers.m+1]
  if product
    q = [prod(q[1:i-1]) for i=1:layers.m+1]
    return q
  else
    return q
  end
end
getQ(layers, Xsize; product=true)=  getQ(layers, Int.(sizes(bspline, layers.subsampling, Xsize[(end):end]))
, [-1 for i=1:layers.m+1]; product=product)
