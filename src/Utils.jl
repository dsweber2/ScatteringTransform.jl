# This file contains mostly random utilities
# TODO: make a way of accessing based on a path, especially in the 1D case
# TODO: make the matrix aggrigator only take the output array

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
    return prod([size(layers.shears[m].shearletIdxs, 1)-1 for k=n:(m-1)])
  end
end

"""
    MatrixAggrigator(pardir::String; keep=[],named="output")

    gathers all sheared matrices in a file with the name grouped(keep) and makes a single matrix out of them, with each row being a single sheared transform. Keep determines which depth to keep, with an empty matrix meaning keep all layers. named is the name of the variable in the file, while it is saved to "grouped(keep)"
"""
function MatrixAggrigator(pardir::String; keep=[], named="output",tyyp::String="Shattering")
  vectorLength = 0
  # First determine the size of a single entry
  for (root, dir, files) in walkdir(pardir)
    # println("folder $root")
    for name in files
      if name[end-3:end]==".jld"
        results = load("$root/$name", named)
        if keep==[]
          if tyyp=="Shattering"
            vectorLength = [prod(size(x)) for x in results.output]
          else
            vectorLength = [prod(size(results))]
          end
        else
          vectorLength = [prod(size(x)) for x in results.output[keep]]
        end
        break
      end
    end
  end
  # quick walkthrough to determine the number of examples
  NumFiles = 0
  for (root, dir, files) in walkdir(pardir)
    for name in files
      if name[end-3:end]==".jld"
        NumFiles+=1
      end
    end
  end
  println("vectorLength = $vectorLength")
  grouped = zeros(NumFiles, sum(vectorLength))
  if keep==[]
    keep = 1:size(vectorLength,1)
  end
  i=1
  # Acutally load the data into the matrix
  for (root, dir, files) in walkdir(pardir)
    for name in files
      if name[end-3:end]==".jld"  && name[1:4]=="data"
        results = load("$root/$name", named)
        if tyyp=="Shattering"
          grouped[i,:] = cat(2,[reshape(results.output[keep[k]],(1,vectorLength[k])) for k=1:size(vectorLength,1)]...)
        else
          grouped[i,:] = reshape(results,(1,vectorLength[1]))
        end
        i+=1
      end
    end
  end
  save("$(pardir)/grouped$(keep).jld","grouped",grouped)
  return grouped
end

# TODO: make a version of this that works for layers other than the last
"""

takes in either a vector that is a linear version of shattered data, or a matrix whose rows are, and reconstructs the appropriately sized tensor from either an example of the output structure,
"""
function reshapeFlattened(mat::Array{Float64,1},layers::layeredTransform)
  n = [[layers.shears[i].size[1] for i=1:(layers.m+1)]; Int64(ceil(layers.shears[end].size[1]/layers.subsampling[end]))] # layers.shears[1].size[1];
  p = [[layers.shears[i].size[2] for i=1:(layers.m+1)]; Int64(ceil(layers.shears[end].size[2]/layers.subsampling[end]))] # layers.shears[1].size[2];
  q = [size(layers.shears[i].shearlets,3) for i=1:layers.m+1]
  outputSize = [(n[i+1], p[i+1], prod(q[1:i-1]-1)) for i=1:layers.m+1]
  return reshape(mat,outputSize[end])
end

function reshapeFlattened(mat::Array{Float64}, sheared::S) where S<:scattered
  return reshape(mat, size(sheared.output[end]))
end

"""

given the (relative) name of a folder and a layered transform, it will load (using a user defined function, which should output data so each column is an example) and transform the entire folder, and save the output into a similarly structured set of folders in destFolder. If separate is true, it will also save a file containing just the scattered coefficients from all inputs.
"""
function transformFolder(sourceFolder::String, destFolder::String, layers::layeredTransform, separate::Bool; loadThis::Function=loadSyntheticMatFile, nonlinear::Function=abs, subsam::Function=bspline, stType::String="full")
  addprocs(14)
  for (root, dirs, files) in walkdir(sourceFolder)
    for dir in dirs
      mkpath(joinpath(destFolder,dir))
    end
    println(root)
    for file in files
      # data is expected as *column* vectors
      (fullMatrix,hasContents) = loadThis(joinpath(root,file))
      if hasContents
        # result = Vector{scattered1D{Complex128}}(size(fullMatrix,2))
        result = Vector{scattered1D{Complex128}}(size(fullMatrix,2))
        @parallel for i = 1:size(fullMatrix,1)
        # for i = 1:size(fullMatrix,1)
          result[i] = st(fullMatrix[i,:],layers, nonlinear=nonlinear, subsam=subsam, stType=stType)
        end
        root[end-+1:end]
        println("saving to $(joinpath(destFolder, relpath(root,sourceFolder),"$(file).jld"))")
        save(joinpath(destFolder, relpath(root,sourceFolder),"$(file).jld"), "result", result)
      end
    end
  end
  println("saving settings to $(joinpath(destFolder,"settings.jld"))")
  save(joinpath(destFolder,"settings.jld"),"layers", layers)
  return
end

function loadSyntheticMatFile(datafile::String)
  hasContents = (datafile[end-3:end]==".mat")
  if hasContents
    wef = matread(datafile)
    return (wef["resp"]', hasContents)
  else
    return ([0.0 0.0;0.0 0.0], hasContents)
  end
end

"""

  given a scattered1D, it produces a single vector containing the entire transform in order, i.e. the same format as output by thinSt
"""
function flatten(results::scattered1D{T}) where T<:Number
  concatOutput = Vector{Float64}(sum([prod(size(x)) for x in results.output]))
  outPos = 1
  for curLayer in results.output
    sizeTmpOut = prod(size(curLayer))
    concatOutput[outPos+(0:sizeTmpOut-1)] = reshape(curLayer, (sizeTmpOut))
    outPos += sizeTmpOut
  end
  concatOutput
end

# 
# """
#
#   given a path p and a flattened scattering transform flat, return the transform of that row.
# """
# function findp(flat::Vector{T}, p::Vector{S}, layers::layeredTransform) where {T <: Number, S<:Integer}
#
