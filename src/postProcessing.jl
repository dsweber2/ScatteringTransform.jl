# This file contains mostly random utilities
# TODO: make a way of accessing based on a path, especially in the 1D case
# TODO: make the matrix aggrigator only take the output array






@doc """
    scattered = wrap(layers::layeredTransform{K, 2}, results::Array{T,N}) where {K, N,
                                                                          T<:
                                                                          Number}

given a layeredTransform and an array as produced by the thin ST, wrap the
results in the easier to process scattered type. note that the data is zero
when produced this way.
"""
function wrap(layers::layeredTransform{K, 2}, results::AbstractArray{T,N}, X;
              percentage=.9) where {K, N, T<: Number}
    wrapped = scattered(layers, X)
    n, q, dataSizes, outputSizes, resultingSize = calculateSizes(layers,
                                                                 (-1,-1),
                                                                 size(X),
                                                                 percentage =
                                                                 percentage)
    for (i,layer) in enumerate(layers.shears[1:layers.m+1])
        concatStart = sum([prod(oneLayer) for oneLayer in
                           outputSizes[1:i-1]]) + 1
        outerAxes = axes(wrapped.output[i])[1:end-3]
        for j = 1:size(wrapped.output[i])[end]
            wrapped.output[i][outerAxes..., :,:,j] = results[outerAxes...,
                                                             concatStart .+
                                                             (j-1)*resultingSize[i] .+
                                                             (0:(resultingSize[i]-1))]
        end
    end
    return wrapped
end
























@doc """
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


function reshapeFlattened(mat::Array{Float64}, sheared::S) where S<:scattered
  return reshape(mat, size(sheared.output[end]))
end

@doc """
    transformFolder(sourceFolder::String, destFolder::String, layers::layeredTransform; separate::Bool=false, loadThis::Function=defaultLoadFunction, nonlinear::Function=abs, subsam::Function=bspline, stType::String="full", thin::Bool=true, postSubsample::Tuple{Int,Int}=(-1,-1), overwrite::Bool=false)

Given the (relative) name of a folder and a layered transform, it will load and transform the entire folder, and save the output into a similarly structured set of folders in destFolder. If separate is true, it will also save a file containing just the scattered coefficients from all inputs.

The data is loaded via a user supplied function loadThis. This should have the signature `(data,hasContents) = loadThis(filename::String)`. `filename` is a (relative) path to the file, while `data` should be arranged so that the  last dimension is to be transformed. `hasContents` is a boolean that indicates if there is actually any data in the file.
If overwrite is false, then skip files that already exist. If overwrite is true, for right now, it uses whatever is already there as the last entries
"""
function transformFolder(sourceFolder::String, destFolder::String, layers::layeredTransform; separate::Bool=false, loadThis::Function=defaultLoadFunction, nonlinear::Function=abs, subsam::Function=bspline, stType::String="full", thin::Bool=true, postSubsample::Tuple{Int,Int}=(-1,-1), overwrite::Bool=false)
  for (root, dirs, files) in walkdir(sourceFolder)
    for dir in dirs
      mkpath(joinpath(destFolder,dir))
    end
    println(root)
    for file in files
      # data is expected as *column* vectors
      println("starting file $(joinpath(root,file))")
      savefile = joinpath(destFolder, relpath(root,sourceFolder), "$(file).h5")
      savefileSize = stat(savefile).size
      if overwrite || savefileSize < 10
        (fullMatrix,hasContents) = loadThis(joinpath(root,file))
        if hasContents
          # breaking up the data into managable chunks
          nEx = size(fullMatrix, 1)
          # if the save file already has meaningful content, load this content
          if savefileSize > 10
            prevWork = h5read(savefile, "result/result")
          end
          # if the savefile doesn't have content, or if there should be more examples than what the save file has, calculate the new coefficients
          if savefileSize < 10 || (nEx>2  && size(prevWork,1)<=2)
            innerAxes = axes(fullMatrix)[2:end]
            usefulSlice = 1:min(2,nEx);
            if thin
              # println("l101")
              tmpResult = thinSt(fullMatrix[usefulSlice, innerAxes...], layers, nonlinear=nonlinear, subsam=subsam, stType=stType, outputSubsample = postSubsample)
            else
              # println("l104")
              tmpResult = st(fullMatrix[usefulSlice, innerAxes...], layers, nonlinear=nonlinear, subsam=subsam, stType=stType)
            end
            # if the data is too big, we'll need extra room to store the results
            result = zeros(nEx, size(tmpResult)[2:end]...)
            innerResultAxes = axes(tmpResult)[2:end]
            result[usefulSlice, innerResultAxes...] = tmpResult
            if nEx>2
              if savefileSize>10
                usefulSlice = (2*ceil(Int,nEx/2)-1):nEx
                result[usefulSlice, innerResultAxes...] = prevWork
              end
              maxValue = (savefileSize>10 && nEx>2) ? ceil(Int,nEx/2)-1 : ceil(Int,nEx/2)
              for i = 2:maxValue
                usefulSlice = (2*i-1):min(nEx, 2*i)
                if thin
                  # println("ln120")
                  tmpResult = thinSt(fullMatrix[usefulSlice, innerAxes...], layers, nonlinear=nonlinear, subsam=subsam, stType=stType, outputSubsample = postSubsample)
                else
                  # println("ln123")
                  tmpResult = st(fullMatrix[usefulSlice, innerAxes...], layers, nonlinear=nonlinear, subsam=subsam, stType=stType)
                end
                # println("ln126")
                result[usefulSlice, innerResultAxes...] = tmpResult
              end
              # repair a problem in the previous run
            end
            println("saving to $(savefile)")
            if savefileSize<10 && isfile(savefile)
              rm(savefile)
            end
            h5write(savefile, "result/result", result)
          end
        end
      end
    end
  end
  println("saving settings to $(joinpath(destFolder,"settings.h5"))")
  save(joinpath(destFolder,"settings.jld"),"layers", layers)
  return
end

function defaultLoadFunction(filename)
  return (h5read(filename,"data"),true)
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

function loadhdf5(datafile::String)
  hasContents = (datafile[end-3:end]==".mat")
  if hasContents
    wef = matread(datafile)
    return (wef["resp"]', hasContents)
  else
    return ([0.0 0.0;0.0 0.0], hasContents)
  end
end

@doc """
flatten(results::scattered{T,1}) where T<:Number
  concatOutput = Vector{Float64}(sum([prod(size(x)) for x in results.output]))

given a scattered, it produces a single vector containing the entire transform in order, i.e. the same format as output by thinSt
"""
function flatten(results::scattered{T,N}, layers::layeredTransform{S, 1}) where {T<:Real, N, S}
  concatOutput = zeros(Float64,size(results.output[1])[1:end-2]..., sum([prod(size(x)[end-1:end]) for x in results.output]))
  outPos = 1
  for curLayer in results.output
    sizeTmpOut = prod(size(curLayer))
    concatOutput[outPos.+(0:sizeTmpOut-1)] = reshape(curLayer, (sizeTmpOut))
    outPos += sizeTmpOut
  end
  concatOutput
end

function flatten(results::scattered{T,N}, layers::layeredTransform{S, 2}) where {T<:Real, N, S}
    concatOutput = zeros(T, size(results.output[1])[1:end-3]...,
                         sum([prod(size(x)[end-2:end]) for x in results.output]))
    outPos = 1
    for curLayer in results.output
        outerAxes = axes(curLayer)[1:end-3]
        sizeTmpOut = prod(size(curLayer)[end-2:end])
        for outer in eachindex(view(curLayer, outerAxes..., 1, 1, 1))
            concatOutput[outer, outPos.+(0:sizeTmpOut-1)] = reshape(curLayer,
                                                                    (sizeTmpOut))
        end
        outPos += sizeTmpOut
    end
    concatOutput
end


# treating them kind of like vectors

import Base.-
-(scattered1::scattered{T,N},scattered2::scattered{S,N}) where {T<:Number,S<:Number, N} = scattered{T, N}(scattered1.m, scattered1.k, [scattered1.data[i] - scattered2.data[i] for i=1:scattered1.m+1], [scattered1.output[i]+ -1*scattered2.output[i] for i=1:scattered1.m+1])

import LinearAlgebra.norm
#norm(scattered1::scattered{T,N},dims::Int=k) where {T<:Number} = sum([norm(scattered.output[i]) for i=1:scattered.m+1])

# function norm(scattered1::scattered{T,N},p::S; dims::Array{Int,1}) where {T<:Number,N,K,S<:Real}
#   setdiff(1:(length(size(scattered1))-1), dims)
#   result = Array{T,N}()
#   reduce(norm, +, view(scattered1),)
#   axes(scattered.output[i])
#   for i=1:
#   sum([norm(view(scattered.output[i][]) for i=1:scattered.m+1]))
# end
"""
   norm(scattered1::scattered{T,N},p::S=2) where {T<:Number, S<:Real, N}

the norm of a scattered result. By default it doesn't take a norm over the leading dimensions
"""
function norm(scattered1::scattered{T,N},p::S=2) where {T<:Number, S<:Real, N}
  results = Array{Float64, N-2}(undef, size(scattered1.output[1])[1:(end-2)])
  outerAxes = axes(scattered1.output[1])[1:end-2]
  for i in eachindex(view(scattered1.output[1], outerAxes..., 1,1))
    results[i] = sum(norm(x[i, :, :], p).^p for x in scattered1.output).^(1/p)
  end
  return results
end
# function norm(scattered1::scattered{T,N},p::S; keepdims::Val{K}) where {T<:Number, N,K}
#   result = Array{T,Val(N) - Val(K)}(undef,size(scattered1))
# function norm(scattered1::scattered{T,N},p::S; dims::Array{Int,1}) where {T<:Number,N,K,S<:Real}
#   result = Array{T, Val(N) - length(dims)}(zeros())
#   return sum(norm(view(x[axes(x)[dims],:]), p).^p for x in scattered1.output)
# end
#wef = randn(10,10,10)
#from scratch version
#(mapreduce(x->mapreduce(y->y.^p, +, x, dims=dims), +, x for x in wef)).^(1/p)
