
struct pathType
  m::Int64
  Idxs
  function pathType(m::Int64,Idxs)
    @assert m==length(Idxs)
    new(m, Idxs)
  end
end

"""
    path = pathType(m::Int64, indexInShear::Array{Int64,1}, layers::layeredTransform)

path constructor that uses the fixed index in each layer (i.e. the index of the shearlet filter in shearletIdxs)
"""
function pathType(m::Int64, indexInLayers::Array{Int64,1}, layers::layeredTransform)
  pathType(m, indexInLayers)
end
"""
    numberSkipped(m::Int64, n::Int64, layers::layeredTransform)
return the number skipped in layer m when moving 1 in layer n
"""
function numberSkipped(m::T, n::T, layers::layeredTransform) where T<:Int
  if m<n
    error("m and n are out of order")
  end
  # no skipping happens in the same layer
  if m==n
    return 1
  end
  if m>n
    return prod([numScales(layers.shears[k], layers.n)-1 for k=n:(m-1)])
  end
end

"""
  initIndex:endIndex = pathToThinIndex(path::pathType, layers::layeredTransform, outputSubsample)
given a path, find the slice in the thinSt that corresponds to that path. Returns the slice to be used with thinStResult, ie thinStResult[metaDims, slice]
"""
function pathToThinIndex(path::pathType, layers::layeredTransform, outputSubsample)
  k, n, q, dataSizes, outputSizes, resultingSize = calculateThinStSizes(layers, outputSubsample, (3, layers.n))
  lengthOfALayer = resultingSize.*q
  endIndex = initIndex = sum(lengthOfALayer[1:(path.m+1)])
  sizes = [layers.n; [outputSizes[j][end-1] for j=1:layers.m]]
  if path.m>0
    for i=2:(path.m+1)
        numChildrenPer = prod([numScales(layers.shears[j], sizes[j])-1 for j=i:path.m+1])
        initIndex += numChildrenPer*(path.Idxs[i-1]-1)*resultingSize[i]
        if i==path.m+1
            endIndex = initIndex + numChildrenPer*resultingSize[i]
            println("added $(numChildrenPer*(path.Idxs[i-1]-1)*resultingSize[i]), ($(numChildrenPer),$((path.Idxs[i-1]-1)), $(resultingSize[i]))")
            println("initIndex = $(initIndex), endIndex = $(endIndex)")
        end
    end
  else
    endIndex = sum(lengthOfALayer[1:(2)])
  end
  return initIndex+1:endIndex
end

"""
  p = computePath(layers::layeredTransform, layerM::Int64, λ::Int64; stType::String="full"))

compute the path p that corresponds to location λ in the flattened transform
"""
function computePath(layers::layeredTransform, λ::Int64; subsampType::String="full")

end


"""
    numChildren = numChildren(path::pathType, layers::layeredTransform)

given a path, determine the number of children it has.
"""
function numChildren(path::pathType, layers::layeredTransform)
  layers.shears[path.m].shearletIdxs
end


"""
  linearInd = getLinearIndex(path::pathType, layers::layeredTransform)

  returns the linear index in the transformed data that corresponds to path.
"""
function getLinearIndex(path::pathType, layers::layeredTransform)
  # linearInd = [ for i=1:path.m]
  # for (i,x) in enumerate(path.Idxs)
  #   for j = 1:numScales(layers.shears[i], layers.n)
  #     if layers.shears[i].shearletIdxs[j,:] == x
  #       linearInd[i] = j
  #     end
  #   end
  # end
  tmpNlayer = [[numScales(layers.shears[j], layers.n)-1 for j=1:path.m]; 1]
  [prod(tmpNlayer[i+1:path.m]) for i=1:path.m]
  path.Idxs'*[prod(tmpNlayer[i+1:path.m]) for i=1:path.m]
end
function getLinearIndex(m::Int64, path::pathType, layers::layeredTransform)
  getLinearIndex(pathType(m,path.Idxs[1:m]),layers)
end



"""
    numInLayer(m::Int64,layers::layeredTransform)

A stupidly straightforward way of counting the number of paths in a given layer.
"""
function numInLayer(m::Int64,layers::layeredTransform)
  nShears = [1; [layers.shears[i].nShearlets-1 for i=1:m]]
  prod(nShears)
end

"""
    (isLast, path) = incrementKeeper(path::Array{Int64}, m::Int64, layers::layeredTransform, nScalesLayers::Array{Int64})
path is a tuple where the ith entry is the [xscale,yscale,shearing] index used at layer i. This function is designed so that it returns the next path. It does so in the order given by shearletIdxs (shearing, xscale, then yscale), and incrementing the deepest layers first. isLast is a boolean which is true if this is the last entry.
"""
function incrementKeeper(path::Array{Int64}, m::Int64, layers::layeredTransform, nScalesLayers::Array{Int64})
  pathInd=1; found = false
  for i = 1:layers.shears[m].nShearlets
    if path.Idxs[m] == layers.shears[m].shearletIdxs[i,:]
      println(i)
      found=true
      break
    end
  end
  if !found
    error("you've created a path that doesn't exist. Specifically, entry $m")
  end
  path.Idxs[m]+=1
  pathInd
  if m==1
    # if we've cycled all possible values, we're done, so return an empty path
    if pathInd >= layers.shears[1].nShearlets
      return (true, Array{Int64}(0))
    else
      path.Idxs[1] =layers.shears[1].shearletIdxs
    end
  else
    # if we've cycled this layer as much as possible, we move down a layer until we hit m==1
    if pathInd >= layers.shears[m].nShearlets
      path.Idxs[m] = layers.shears[m].shearletIdxs[1,:]
      return incrementKeeper(path, m-1, scalingFactors, nScalesLayers)
    else
      path.Idxs[m] =layers.shears[m].shearletIdxs
    end
  end
  return (false, path)
end
# """
#     (isLast, keeper) = incrementKeeper(keeper::Array{Float64}, m::Int, resolutions::Array{Float64})
# keeper is a tuple where the ith entry is the scale used at layer i. This function is designed so that it only returns decreasing entries (including effects from the scales per octave). It does so in order from smallest scale in all layers, to largest in all layers, incrementing the deepest layers first. isLast is a boolean which is true if this is the last entry.
# """
# function incrementKeeper(keeper::Array{Int}, m::Int, scalingFactors::Array{Float64,1}, nScalesLayers::Array{Int})
#   keeper[m]+=1
#   if m==1
#     # if we've cycled all possible values, we're done, so return an empty keeper
#     if keeper[1] > nScalesLayers[1]
#       return (true, Array{Int}(0))
#     end
#   else
#     # if we've cycled this layer as much as possible, we move on
#     if keeper[m] > nScalesLayers[m] || floor(keeper[m-1] ./scalingFactors[m-1]) < ceil(keeper[m] ./scalingFactors[m])
#       keeper[m]=1
#       return incrementKeeper(keeper, m-1, scalingFactors, nScalesLayers)
#     end
#   end
#   return (false, keeper)
# end
# incrementKeeper(keeper::Array{Int}, m::Int, layers::layeredTransform, nScalesLayers::Array{Int}) = incrementKeeper(keeper, m, [shear.scalingFactor for shear in layers.shears], nScalesLayers::Array{Int})
