# Actual scattering function
########################################################
########################################################
# TODO: order shouldn't matter since the types are unique










# TODO: make a function which determines the breakpoints given the layers function and the size of the input
@doc """
        st(X::Array{T}, layers::layeredTransform; <keyword arguments>nonlinear::Function=abs, subsam::Function=bspline, stType::String="full", outputSubsample::Tuple{Int,Int}=(-1,1), totalScales = [-1 for i=1:layers.m+1]) where {T<:Real}

      See the main st for a description of the options. This is a version of the 1D st that only returns a concatinated vector for output. It is most useful for classification applications.

    outputSubsample is a tuple, with the first number indicating a rate, and the second number giving the minimum allowed size. If some of the entries are less than 1, it has different behaviour:
    (<1, x): subsample to x elements for each path.
    (<1, <1): no ssubsampling
    (x, <1) subsample at a rate of x, with at least one element kept in each path

    totalScales, if positive, gives the number of non-averaging wavelets.
TODO: non-thin is currently broken
  1D scattering transform using the layeredTransform layers. you can switch out the nonlinearity as well as the method of subsampling. Finally, the stType is a string. If it is "full", it will produce all paths. If it is "decreasing", it will only keep paths of increasing scale. If it is "collating", you must also include a vector of matrices.
# Arguments
- `fullOr::fullType=fullType()` : the structure of the transform either
       `fullType()`, `collatingType()` or `decreasingType()`. At the moment,
       only `fullType()` is functional.
- `fftPlans = false` if not `false`, it should be a 2D array of `Future`s,
where the x index gives the

 st(X::Array{T, N}, layers::layeredTransform,
            nonlinear::nl, fullOr::fullType=fullType();
            thin::Bool=false, subsam::Bool=true,
            totalScales = [-1 for i=1:layers.m+1]) where {T<:Real,
                                                          S<:Union, N, nl <:
                                                          nonlinearity}
    """
function st(X::Array{T, N}, layers::layeredTransform, nonlinear::nl;
            fullOr::fullType=fullType(),# subsam::Sub = bspline(), #TODO
            # allow for variation in subsamping method again
            thin::Bool=true, outputSubsample::Tuple{Int, Int}=(-1,-1),
            subsam::Bool=true, totalScales = [-1 for i=1:layers.m+1],
            percentage = .9, fftPlans = -1, verbose::Bool=false) where {T <: Real, S <: Union, N, nl <:
                                    nonlinearity, Sub <: resamplingMethod}
    @assert length(totalScales)==layers.m+1
    @assert fftPlans ==-1 || typeof(fftPlans)<:Array{<:Future,2}
    if verbose
        println("verbose mode on. This may have more output than you'd like to put up with")
    end
    dataDim = eltypes(layers)[2]
    # Insist that X has to have at least one extra meta-dimension, even if it
    # is 1
    if length(size(X)) == dataDim
        X = reshape(X, (1,size(X)...));
    end
    numChildλ = 0
    n, q, dataSizes, outputSizes, resultingSize = calculateSizes(layers,
                                                                 outputSubsample,
                                                                 size(X),
                                                                 totalScales =
                                                                 totalScales,
                                                                 percentage =
                                                                 percentage)
    nextData = [reshape(X, (size(X)..., 1))]
    if verbose
        println("about to make the storage arrays")
    end
    # get the size of a single entry as a long string
    if dataDim==1 && outputSubsample[1] > 1 || outputSubsample[2] > 1
        netSize = Int.(sum(resultingSize.*q))
    else
        netSize = Int(sum([prod(x[end-dataDim:end]) for x in outputSizes])...)
    end
    # create list of size references if we're subsampling the output an
    # extra amount
    if dataDim==1
        println("sum(q .* resultingSize)) = $(sum(q .* resultingSize))), 
            resultingSize = $(resultingSize), q=$(q)")
        results = SharedArray(zeros(T, outputSizes[1][1:end-dataDim-1]..., 
                                    sum(q .* resultingSize)))
    else
        results = SharedArray(zeros(T, size(X)[1:end-dataDim]...,
                                    sum(prod(oneLayer[end-2:end]) for oneLayer in
                                        outputSizes)...))
    end
    if verbose
        println("made the storage arrays, about to make the fft plans")
    end
    if fftPlans == -1 && verbose
        @time fftPlans = createFFTPlans(layers, dataSizes, verbose = verbose, iscomplex = T<:Complex)
    elseif fftPlans == -1
        fftPlans = createFFTPlans(layers, dataSizes, T=T, iscomplex = T<:Complex)
    end
    if verbose
        println("about to start iterating")
    end 
    results = iterateOverLayers!(layers, results, nextData, dataSizes,
                                 outputSizes, dataDim, q, totalScales, T, thin,
                                 nonlinear, subsam, outputSubsample,
                                 resultingSize, fftPlans, verbose)
    if verbose
        println("finished iterate Over layers")
    end
    if thin
        return results
    else
        return wrap(layers, results, X, percentage = percentage)
    end
end
function st(layers::layeredTransform, X::Array{T, N}, nonlinear::nl;
            fullOr::fullType=fullType(),# subsam::Sub = bspline(), #TODO
            # allow for variation in subsamping method again
            thin::Bool=true, outputSubsample::Tuple{Int, Int}=(-1,-1),
            subsam::Bool=true, totalScales = [-1 for i=1:layers.m+1],
            percentage = .9, fftPlans = -1) where {T <: Real, S <: Union, N, nl <:
                                    nonlinearity, Sub <: resamplingMethod}
    return st(X, layers, nonlinear, fullOr, thin = thin,
              outputSubsample = outputSubsample, subsam = subsam, totalScales =
              totalScales, percentage = percentage)
end
st(layers::layeredTransform, X::Array{T}; nonlinear::Function=abs, subsam::Function=bspline) where {T<:Real} = st(X,layers,nonlinear=nonlinear, subsamp=subsam,verbose=false)


function iterateOverLayers!(layers, results, nextData, dataSizes, outputSizes,
                           dataDim, q, totalScales, T, thin, nonlinear, subsam,
                           outputSubsample, resultingSize, fftPlans,verbose)
    for (i,layer) in enumerate(layers.shears[1:layers.m])
        if verbose
            println("On layer $(i)")
        end 
        cur = nextData #data from the previous layer
        # only store the intermediate results in intermediate layers
        if i < layers.m
            # store the data in a channel, along with the number of
            # it's parent
            dataChannel = Channel{ Tuple{ Array{ length(dataSizes[i+1])},
                                          Int64}}(size(cur)[end])
            nextData = Array{SharedArray{T, length(dataSizes[i+1])},
                             1}(undef, size(cur[1])[end]*length(cur))
        end

        innerAxes = axes(cur[1])[end-dataDim:end-1] # effectively a
        # set of colons of length dataDim, to be used for input
        
        innerSub = (Base.OneTo(x) for x in
                    dataSizes[i+1][end-dataDim:end-1]) # effectively a
        # set of colons of length dataDim, to be used for the
        # subsampled output

        outerAxes = axes(cur[1])[1:(end-dataDim-1)] # the same idea as
        # innerAxes, but for the example indices

        # precompute the wavelets and the fft plan
        if dataDim == 1
            daughters = computeWavelets(size(cur[1])[end-1], layers.shears[i],
                                        nScales=totalScales[i])
            concatStart = reduce(+,
                                 q[1:i-1] .* resultingSize[1:i-1],
                                 init = 0) + 1
            # create an array for the fftPlan
            padBy = (0, 0)
        else
            concatStart = sum([prod(oneLayer[end-2:end]) for oneLayer in
                               outputSizes[1:i-1]])+1
            daughters = []
            padBy = getPadBy(layers.shears[i])
        end

        println("q =  $(q), resultingSize = $(resultingSize), size(d) = $(size(daughters))")
        listOfProcessResults = Array{Future, 1}(undef,
                                                size(cur[1])[end]*length(cur))
        # parallel processing tools; the results are spread over the previous 2
        # layers paths

        if verbose
            println("about to send to spawningJobs!")
        end 
        # iterate over the scales two layers back
        if i==layers.m
            # compute the final mother if we're on the last layer
            if dataDim==1
                finalDaughters = computeWavelets(dataSizes[end][end-1],
                                         layers.shears[i+1], nScales=1)
                concatStartLast = reduce(+, (q .* resultingSize)[1:end-1], init = 0) + 1
                println("concatStartLast = $(concatStartLast)")
                padByLast = (1,3)
                nScalesLastLayer = size(daughters,2)-1
            else
                mother = []
                finalDaughters = []
                concatStartLast = sum([prod(oneLayer[end-2:end]) for oneLayer
                                       in outputSizes[1:end-1]]) + 1
                padByLast = getPadBy(layers.shears[end])
                nScalesLastLayer = q[end]
            end
            spawningJobs!(listOfProcessResults, layers, results, cur,
                          outerAxes, innerAxes, innerSub, dataDim, i,
                          daughters, finalDaughters, totalScales[i], nonlinear,
                          subsam, outputSubsample, outputSizes, resultingSize,
                          concatStart, concatStartLast, padBy, padByLast,
                          nScalesLastLayer, fftPlans, T, dataSizes,verbose)
        else 
            # any layer before the penultimate
            spawningJobs!(listOfProcessResults, layers, results, cur,
                          dataSizes, outerAxes,innerAxes, innerSub, dataDim, i,
                          daughters, totalScales[i], nonlinear, subsam,
                          outputSubsample, outputSizes[i], concatStart, padBy,
                          resultingSize, dataChannel, fftPlans, T,verbose)
        end
        # using one channel per
        for (λ,x) in enumerate(listOfProcessResults)
            tmpFetch = fetch(x)
            if typeof(tmpFetch) <: Exception
                throw(tmpFetch)
            end
            if i<layers.m
                nextData[λ] = tmpFetch
            end
        end
        if verbose
            println("fetched from workers")
        end
    end
    results
end

# the midlayer version, i.e. i≠m
function spawningJobs!(listOfProcessResults, layers, results, cur, dataSizes,
                       outerAxes, innerAxes, innerSub, dataDim, i, daughters,
                       totalScale, nonlinear, subsam, outputSubsample,
                       outputSize, concatStart, padBy, resultingSize,
                       dataChannel, fftPlans, T, verbose)
    njobs = Distributed.nworkers();
    cjob = 0;
    for (j,x) in enumerate(cur)
        # iterate over all paths in layer i
        if verbose
            #println("On path $j")
        end 
        for λ = 1:size(x)[end]
            # do the actual transformation; the last layer requires
            # far less saved than the mid-layers. cjob+1 is the worker to
            # assign the task to (+1 since mod is 0...n-1 rather than 1...n
            if verbose
                #println("On sub-path $λ, sending to worker $cjob")
            end
            futureThing = remotecall(midLayer!, cjob+1, layers, results, x,
                                     dataSizes, outerAxes, innerAxes, innerSub,
                                     dataDim, i, daughters, totalScale,
                                     nonlinear, subsam, outputSubsample,
                                     outputSize, concatStart, padBy, λ,
                                     resultingSize, dataChannel,
                                     fftPlans[cjob+1, i], T)
            cjob = mod(cjob+1, njobs)
            if typeof(futureThing) <: typeof(Distributed.RemoteException)
                throw(futureThing)
            else
                listOfProcessResults[(j-1)*size(cur[1])[end] + λ] = futureThing
            end
        end
    end
end

# the end version, i.e. i=m

function spawningJobs!(listOfProcessResults, layers, results, cur, outerAxes,
                       innerAxes, innerSub, dataDim, i, daughters,
                       finalDaughters, totalScale, nonlinear, subsam,
                       outputSubsample, outputSizes, resultingSize,
                       concatStart, concatStartLast, padBy, padByLast,
                       nScalesLastLayer, fftPlans, T, dataSizes, verbose)
    njobs = Distributed.nworkers();
    cjob = 0;
    for (j,x) in enumerate(cur)
        # iterate over all paths in layer i
        for λ = 1:size(x)[end]
            # do the actual transformation; the last layer requires
            # far less saved than the mid-layers

            futureThing = remotecall(finalLayer!, cjob+1, layers, results, x,
                                     outerAxes, innerAxes, innerSub, dataSizes,
                                     dataDim, i, daughters, finalDaughters,
                                     totalScale, nonlinear, subsam,
                                     outputSubsample, outputSizes, λ,
                                     resultingSize, concatStart,
                                     concatStartLast, padBy, padByLast,
                                     nScalesLastLayer, fftPlans[cjob+1,
                                     i],fftPlans[cjob+1, i+1], T)

            cjob = mod(cjob+1, njobs)
            if typeof(futureThing)<: typeof(Distributed.RemoteException)
                throw(futureThing)
            else
                listOfProcessResults[(j-1)*size(cur[1])[end] + λ] = futureThing
            end

        end
    end
    return listOfProcessResults
end


@doc """
        midLayer!(layers, results, cur, dataSize, outerAxes, innerAxes, innerSub, dataDim, i, daughters, nScales, nonlinear, subsam, outputSubsample, concatStart, λ)
        midLayer!(layers, results, cur, nextData, outerAxes, innerAxes, innerSub, outerAxes, dataDim, i, nScalesLayers, daughters, nScales, nonlinear, subsam, outputSubsample,
     λ, keeper)
    An intermediate function which takes a single path in a layer and generates all of the children of that path in the next layer. Only to be used on intermediate layers, as finalLayerTransform fills the same role in the last layer. If keeper is omitted, it assumes we don't have decreasing type, otherwise it's assumed

"""

function midLayer!(layers::layeredTransform{K,1}, results, curPath, dataSizes,
                   outerAxes, innerAxes, innerSub, dataDim, i, daughters,
                   nScales, nonlinear, subsam, outputSubsample,
                   outputSizeThisLayer, concatStart, padBy, λ, resultingSize,
                   homeChannel, fftPlan, T) where {K}

    # make an array to return the results
    toBeHandedBack = zeros(T, dataSizes[i+1][1:end-1]..., size(daughters,2)-1)
    # first perform the continuous wavelet transform on the data from the
    # previous layer 
    println("$(size(curPath)), $((outerAxes)), inner = $(innerAxes), λ = $(λ)")
    output = cwt(view(curPath, outerAxes..., innerAxes..., λ), layers.shears[i],
                 daughters, fftPlan, nScales=nScales)
    innerMostAxes = axes(output)[end:end]
    # iterate over the non transformed dimensions of output
    writeLoop!(output, outerAxes, i, T,dataSizes, innerAxes, innerSub, layers,
               results, toBeHandedBack, concatStart,λ, dataDim,outputSize,
               nonlinear, subsam, outputSizeThisLayer, outputSubsample,
               resultingSize)

    return toBeHandedBack
end


function midLayer!(layers::layeredTransform{K,2}, results, curPath, dataSizes,
                   outerAxes, innerAxes, innerSub, dataDim, i, daughters,
                   nScales, nonlinear, subsam, outputSubsample,
                   outputSizeThisLayer, concatStart, padBy, λ, resultingSize,
                   homeChannel, fftPlan, T) where {K}
    # make an array to return the results
    toBeHandedBack = zeros(T, dataSizes[i+1][1:end-1]...,
                           size(layers.shears[i].shearlets)[end] - 1)
    # first perform the continuous wavelet transform on the data from the
    # previous layer
    output = sheardec2D(view(curPath, outerAxes..., innerAxes..., λ),
                        layers.shears[i], fftPlan, true, padBy)
    innerMostAxes = axes(output)[end:end]
    # iterate over the non transformed dimensions of output
    writeLoop!(output, outerAxes, i, T,dataSizes, innerAxes, innerSub, layers, results,
               toBeHandedBack,concatStart,λ, dataDim,outputSize, nonlinear,
               subsam,outputSizeThisLayer, outputSubsample, resultingSize)
    return toBeHandedBack
end

for nl in functionTypeTuple
    @eval begin
        function writeLoop!(output, outerAxes, i, T,dataSizes, innerAxes, innerSub, layers,
                            results, toBeHandedBack, concatStart, λ, dataDim,
                            outputSize, nonlinear::$(nl[2]), subsam,
                            outputSizeThisLayer, outputSubsample,
                            resultingSize)
            for outer in eachindex(view(output, outerAxes..., [1 for
                                                               i=1:(dataDim+1)]...))
                # output may have length zero if the starting path has low
                # enough scale
                for j = 2:size(output)[end]
                    toBeHandedBack[outer, innerSub..., j-1] =
                        $(nl[1]).(resample(view(output, outer, innerAxes...,
                                                j), layers.subsampling[i]))
                    #println("λ=$(λ), tbh")
                end
                # actually write to the output
                if subsam
                    tmpOut = Array{T, dataDim}(real.(resample(view(output,
                                                                   outer,
                                                                   innerAxes...,
                                                                   size(output)[end]),
                                                              0f0, newSize =
                                                              outputSizeThisLayer[end-dataDim:end-1])))
                else
                    tmpOut = Array{T, dataDim}(real(resample(view(output,
                                                                  outer,
                                                                  innerAxes...,
                                                                  size(output)[end]),
                                                             layers.subsampling[i])))
                end
                sizeTmpOut = prod(size(tmpOut))
                if outputSubsample[1] > 1 || outputSubsample[2] > 1
                    results[outer, concatStart .+ (λ-1)*resultingSize[i] .+
                            (0:(resultingSize[i]-1))] = resample(tmpOut, 0f0,
                                                                 newSize =
                                                                 resultingSize[i])
                else
                    results[outer, concatStart .+ (λ-1)*resultingSize[i] .+
                            (0:sizeTmpOut-1)] = reshape(tmpOut, (sizeTmpOut))
                end
            end
        end


@doc """
        finalLayer!(layers, results, curPath, outerAxes, innerAxes, innerSub, dataDim, i, daughters, nScales, nonlinear, subsam, outputSubsample, λ, resultingSize, concatStart, λ)

    Do the transformation of the m-1 to mth layer, the output of the m-1st layer, and the output of the mth layer. These are glued together for mostly for computational reasons. There is currently no decreasing paths version, so it should end up throwing errors
    """
function finalLayer!(layers::layeredTransform{K,1}, results, curPath,
                     outerAxes, innerAxes, innerSub, dataSizes, dataDim, i,
                     daughters, finalDaughters, nScales, nonlinear::$(nl[2]),
                     subsam, outputSubsample, outputSizes, λ, resultingSize,
                     concatStart, concatStartLast, padBy, padByLast,
                     nScalesLastLayer, fftPlan, fftPlanFinal, T) where {K}
    localIndex = 0
    # iterate over the outerAxes
    # first perform the continuous wavelet transform on the data from the
    # previous layer
    output = cwt(view(curPath, outerAxes..., innerAxes..., λ),
                 layers.shears[i], daughters, fftPlan, nScales=nScales)
    toBeHandedOnwards = zeros(T, dataSizes[end][1:end-1]...,
                              size(daughters,2) - 1)
    writeLoop!(output, outerAxes, i, T, dataSizes, innerAxes, innerSub, layers,
               results, toBeHandedOnwards, concatStart, λ, dataDim, outputSize,
               nonlinear, subsam, outputSizes[end-1], outputSubsample,
               resultingSize)


    innerAxes = axes(toBeHandedOnwards)[end-1]
    outerAxes = axes(toBeHandedOnwards)[1:end-2]
    # write the output from the mth layer to output
    println("size(output) = $(size(output))")
    for j = 2:size(output)[end]
        finalOutput = cwt(view(toBeHandedOnwards, outerAxes..., innerAxes, j-1),
                          layers.shears[i+1], finalDaughters, fftPlanFinal,
                          nScales=1)
        println("λ = $(λ), j = $(j), $(size(finalOutput))")
        for outer in eachindex(view(finalOutput, outerAxes..., [1 for
                                                                i=1:(dataDim+1)]...))
            tmpFinal = Array{Float64, dataDim}(real.(resample(finalOutput[outer,
                                                                          :],
                                                              0f0,
                                                              newSize =
                                                              resultingSize[end])))
            if outputSubsample[1] > 1 || outputSubsample[2] > 1
                results[outer, concatStartLast +
                        (λ-1)*nScalesLastLayer*resultingSize[end] +
                        (j-2)*resultingSize[i+1] .+ (0:(resultingSize[i+1]-1))] =
                        resample(tmpFinal, 0f0, newSize = resultingSize[i+1])
            else
                sizeTmpOut = prod(size(tmpFinal))
                println("sizeTmp out = $(sizeTmpOut)")
                results[outer, (concatStartLast +
                                (λ-1)*nScalesLastLayer*resultingSize[end] +
                                (j-2)*resultingSize[i+1]).+(0:(sizeTmpOut-1))] = reshape(tmpFinal, (sizeTmpOut))
            end
        end
    end
end
end
end
@doc """
        finalLayer!(layers, results, curPath, outerAxes, innerAxes, innerSub, dataDim, i, daughters, nScales, nonlinear, subsam, outputSubsample, λ, resultingSize, concatStart, λ)

    Do the transformation of the m-1 to mth layer, the output of the m-1st layer, and the output of the mth layer. These are glued together for mostly for computational reasons. There is currently no decreasing paths version, so it should end up throwing errors
    """
function finalLayer!(layers::layeredTransform{K, 2}, results, curPath,
                     outerAxes, innerAxes, innerSub, dataSizes, dataDim, i,
                     daughters, nScales, nonlinear, subsam, outputSubsample,
                     outputSizes, λ, resultingSize, concatStart,
                     concatStartLast, padBy, padByLast, nScalesLastLayer,
                     fftPlan, fftPlanFinal, T) where {K} 
    localIndex = 0
    # iterate over the outerAxes
    # first perform the continuous wavelet transform on the data from the
    # previous layer
    output = sheardec2D(view(curPath, outerAxes..., innerAxes..., λ),
                        layers.shears[i], fftPlan, true, padBy) 
    toBeHandedOnwards = zeros(T, dataSizes[end][1:end-1]...,
                              size(layers.shears[i].shearlets)[end] - 1)
    # store for the final output
    writeLoop!(output, outerAxes, i, T, dataSizes, innerAxes, innerSub, layers,
               results, toBeHandedOnwards, concatStart, λ, dataDim, outputSize,
               nonlinear, subsam, outputSizes[end-1], outputSubsample,
               resultingSize)

    # write the output from the mth layer to output
    for j = 2:size(output)[end]
        for outer in eachindex(view(output, outerAxes..., [1 for
                                                           i=1:(dataDim+1)]...))
            finalOutput = averagingFunction(view(toBeHandedOnwards, outer,
                                                 innerSub..., j-1),
                                            layers.shears[i+1], fftPlanFinal,
                                            true, padByLast)
            tmpFinal = Array{Float64,dataDim}(real.(resample(finalOutput, 0f0,
                                                             newSize =
                                                             outputSizes[end][end-2:end-1])))
            if outputSubsample[1] > 1 || outputSubsample[2] > 1
                results[outer, concatStartLast +
                        (λ-1)*nScalesLastLayer*resultingSize[i] +
                        (j-2)*resultingSize[i] .+ (0:(resultingSize[i+1]-1))] =
                        resample(tmpFinal, outputSizes[i+1][2])
            else
                sizeTmpOut = prod(size(tmpFinal))
                results[outer, (concatStartLast +
                                (λ-1)*nScalesLastLayer*resultingSize[end] +
                                localIndex +
                                (j-2)*resultingSize[i+1]).+(0:(sizeTmpOut-1))] = reshape(tmpFinal, (sizeTmpOut))
            end
        end
    end
end
