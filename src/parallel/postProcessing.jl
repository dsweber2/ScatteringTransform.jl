# This file contains mostly random utilities
# TODO: make the matrix aggrigator only take the output array






@doc """
    Scattered = wrap(layers::stParallel{K, M}, results::Array{T,N}, X) where {K, N, M, T<:Number}

given a stParallel and an array as produced by the thin ST, wrap the
results in the easier to process Scattered type. note that the data is zero
when produced this way.

TODO: the 2nd layer still has a weird size, since both indices are stuck together (see the cwt example)
"""
function roll(layers::stParallel{K, 1}, results::AbstractArray{T,N}, X;
              percentage=.9, outputSubsample=(-1,-1),full=false) where {K, N, T<: Number}
    if full
        wrapped = ScatteredFull(layers, X, outputSubsample=outputSubsample)
    else
        wrapped = ScatteredOut(layers, X, outputSubsample=outputSubsample)
    end
    n, q, dataSizes, outputSizes, resultingSize = calculateSizes(layers,
                                                                 outputSubsample,
                                                                 size(X),
                                                                 percentage =
                                                                 percentage)
    @info "" n' q dataSizes outputSizes resultingSize
    outerAxes = axes(X)[2:end]
    println(outerAxes)
    addedLayers = parallel.getListOfScaleDims(layers,n)
    for i = 1:depth(layers)+1
        concatStart = sum([prod(oneLayer[1:end-ndims(X)+1]) for oneLayer in
                           outputSizes[1:i-1]]) + 1
        println(outputSizes[1:i-1],concatStart)
        for js in indexOverScales(addedLayers,i)
            j = parallel.linearFromTuple(js, addedLayers[i])
            if i==3
                println("js=$js, j=$j")
                println(concatStart .+ j*resultingSize[i] .+ (0:(resultingSize[i]-1)))
                println(j*resultingSize[i])
            end
            accessed = results[concatStart .+
                                           j*resultingSize[i] .+
                                           (0:(resultingSize[i]-1)),
                               outerAxes...]
            thingToWrite = reshape(accessed, 
                                   (resultingSize[i],
                                    size(X)[2:end]...,))
            if js == 1
                wrapped.output[i][:, outerAxes...] = thingToWrite
            else
                wrapped.output[i][:,js..., outerAxes...] = thingToWrite
            end
        end
    end
    return wrapped
end


















@doc """
    MatrixAggrigator(pardir::String; keep=[],named="output")

    gathers all sheared matrices in a file with the name grouped(keep) and makes a single matrix out of them, with each row being a single sheared transform. Keep determines which depth to keep, with an empty matrix meaning keep all layers. named is the name of the variable in the file, while it is saved to "grouped(keep)"
"""
function MatrixAggrigator(pardir::String; keep=[], named="output")
  vectorLength = 0
  # First determine the size of a single entry
  for (root, dir, files) in walkdir(pardir)
    @debug "folder $root"
    for name in files
      if name[end-3:end]==".jld"
        results = load("$root/$name", named)
        if keep==[]
            vectorLength = [prod(size(results))]
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
  @debug "vectorLength = $vectorLength"
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
          grouped[i,:] = reshape(results,(1,vectorLength[1]))
          i+=1
      end
    end
  end
  save("$(pardir)/grouped$(keep).jld","grouped",grouped)
  return grouped
end


function reshapeFlattened(mat::Array{Float64}, sheared::S) where S<:Scattered
  return reshape(mat, size(sheared.output[end]))
end

@doc """
    transformFolder(sourceFolder::String, destFolder::String, layers::stParallel; separate::Bool=false, loadThis::Function=defaultLoadFunction, nonlinear::Function=abs, subsam::Function=bspline, stType::String="full", thin::Bool=true, postSubsample::Tuple{Int,Int}=(-1,-1), overwrite::Bool=false)

Given the (relative) name of a folder and a layered transform, it will load and transform the entire folder, and save the output into a similarly structured set of folders in destFolder. If separate is true, it will also save a file containing just the Scattered coefficients from all inputs.

The data is loaded via a user supplied function loadThis. This should have the signature `(data,hasContents) = loadThis(filename::String)`. `filename` is a (relative) path to the file, while `data` should be arranged so that the  last dimension is to be transformed. `hasContents` is a boolean that indicates if there is actually any data in the file.
If overwrite is false, then skip files that already exist. If overwrite is true, for right now, it uses whatever is already there as the last entries
"""
function transformFolder(sourceFolder::String, destFolder::String, layers::stParallel; separate::Bool=false, loadThis::Function=defaultLoadFunction, nonlinear::Function=abs, subsam::Function=bspline, stType::String="full", thin::Bool=true, postSubsample::Tuple{Int,Int}=(-1,-1), overwrite::Bool=false)
  for (root, dirs, files) in walkdir(sourceFolder)
    for dir in dirs
      mkpath(joinpath(destFolder,dir))
    end
    @debug "" root
    for file in files
      # data is expected as *column* vectors
      @debug "starting file $(joinpath(root,file))"
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
              tmpResult = thinSt(fullMatrix[usefulSlice, innerAxes...], layers, nonlinear=nonlinear, subsam=subsam, stType=stType, outputSubsample = postSubsample)
            else
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
                  tmpResult = thinSt(fullMatrix[usefulSlice, innerAxes...], layers, nonlinear=nonlinear, subsam=subsam, stType=stType, outputSubsample = postSubsample)
                else
                  tmpResult = st(fullMatrix[usefulSlice, innerAxes...], layers, nonlinear=nonlinear, subsam=subsam, stType=stType)
                end
                result[usefulSlice, innerResultAxes...] = tmpResult
              end
              # repair a problem in the previous run
            end
            @info "saving to $(savefile)"
            if savefileSize<10 && isfile(savefile)
              rm(savefile)
            end
            h5write(savefile, "result/result", result)
          end
        end
      end
    end
  end
  @info "saving settings to $(joinpath(destFolder,"settings.h5"))"
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

