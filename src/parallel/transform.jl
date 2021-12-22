# Actual scattering function
########################################################
########################################################
# TODO: order shouldn't matter since the types are unique










# TODO: make a function which determines the breakpoints given the layers function and the size of the input
# TODO: non-thin is currently not outputing values in the data section
@doc """
see the docs for stParallel
"""
function st(X::Array{T,N}, layers::stParallel, nonlinear::nl;
    fullOr::fullType = fullType(),# subsam::Sub = bspline(), #TODO
    # allow for variation in subsamping method again
    thin::Bool = true, outputSubsample::Tuple{Int,Int} = (-1, -1),
    subsam::Bool = true, totalScales = [-1 for i = 1:depth(layers)+1],
    percentage = 0.9, fftPlans = -1) where {T<:Real,S<:Union,N,nl<:
    Function,Sub<:resamplingMethod}
    @assert length(totalScales) == depth(layers) + 1
    @assert fftPlans == -1 || typeof(fftPlans) <: Array{<:Future,2}
    if T <: Float64
        @warn "data type is $T. This is probably higher precision than you need and likely to eat a lot of memory"
    end
    dataDim = eltypes(layers)[2]
    # Insist that X has to have at least one extra meta-dimension, even if it is 1
    if ndims(X) == dataDim
        X = reshape(X, (size(X)..., 1))
    end
    numChildλ = 0
    n, q, dataSizes, outputSizes, resultingSize =
        calculateSizes(layers, outputSubsample, size(X), totalScales = totalScales, percentage = percentage)

    nextData = [reshape(X, (size(X)[1:dataDim]..., 1, size(X)[(dataDim+1):end]...))]
    @debug "about to make the storage arrays"

    # get the size of a single entry as a single vector (outputSizes gives the
    # # of paths)
    netSize = Int(sum([resultingSize[i] * x[dataDim+1] for (i, x) in
                       enumerate(outputSizes)])...)
    @debug "netSize = $(netSize), resultingSize=$(resultingSize), outputSizes=$(outputSizes)"

    # create list of size references if we're subsampling the output an
    # extra amount
    results = SharedArray(zeros(T, netSize, outputSizes[1][dataDim+2:end]...))
    @debug "outputSizes = $(outputSizes)"
    @debug "made the storage arrays, about to make the fft plans"

    if fftPlans == -1
        @debug "dataSizes = $(dataSizes)"
        fftPlans = createFFTPlans(layers, dataSizes, T = T, iscomplex = T <: Complex)
    end
    @debug "about to start iterating"
    results = iterateOverLayers!(layers, results, nextData, dataSizes,
        outputSizes, dataDim, q, totalScales, T, thin,
        nonlinear, subsam, outputSubsample,
        resultingSize, fftPlans)
    @debug "finished iterate Over layers"


    if thin
        return results
    else
        return roll(layers, results, X; percentage = percentage,
            outputSubsample = outputSubsample)
    end
end
function st(layers::stParallel, X::Array{T,N}, nonlinear::nl;
    fullOr::fullType = fullType(),# subsam::Sub = bspline(), #TODO
    # allow for variation in subsamping method again
    thin::Bool = true, outputSubsample::Tuple{Int,Int} = (-1, -1),
    subsam::Bool = true, totalScales = [-1 for i = 1:depth(layers)+1],
    percentage = 0.9, fftPlans = -1) where {T<:Real,S<:Union,N,nl<:
    Function,Sub<:resamplingMethod}
    return st(X, layers, nonlinear, fullOr, thin = thin,
        outputSubsample = outputSubsample, subsam = subsam, totalScales = totalScales, percentage = percentage)
end
st(layers::stParallel, X::Array{T}; nonlinear::Function = abs, subsam::Function = bspline) where {T<:Real} = st(X, layers, nonlinear = nonlinear, subsamp = subsam)


function iterateOverLayers!(layers, results, nextData, dataSizes, outputSizes,
    dataDim, q, totalScales, T, thin, nonlinear, subsam,
    outputSubsample, resultingSize, fftPlans)
    for (i, layer) in enumerate(layers.shears[1:depth(layers)])
        @debug "On layer $(i)"
        @debug "size(nextData) = $(size(nextData[1]))"
        cur = nextData # data from the previous layer

        # only store the intermediate results in intermediate layers
        # TODO: get rid of the data channel, or replace nextData with it
        if i < depth(layers)
            # store the data in a channel, along with the number of
            # it's parent
            dataChannel = Channel{Tuple{Array{length(dataSizes[i+1])},Int64}}(size(cur)[end])
            filler = SharedArray(zeros(T, [1 for i = 1:length(dataSizes[1])]...))
            nextData = SharedArray[filler for i = 1:size(cur[1])[dataDim+1]*length(cur)]
        end

        @debug "cur = $(typeof(cur)), $(size(cur))"
        innerAxes = axes(cur[1])[1:dataDim] # effectively a
        # set of colons of length dataDim, to be used for input
        @debug "innerAxes= $(innerAxes)"
        innerSub = (Base.OneTo(x) for x in
                    dataSizes[i+1][1:dataDim]) # effectively a
        # set of colons of length dataDim, to be used for the
        # subsampled output

        outerAxes = axes(cur[1])[(dataDim+2):end] # the same idea as
        # innerAxes, but for the example indices
        @debug "outerAxes= $(outerAxes)"
        wev = 3
        # precompute the wavelets and the fft plan
        if dataDim == 1
            @debug "earlier layer $(i)"
            @debug "size(cur[1]) = $(size(cur[1]))"
            daughters, ω = computeWavelets(size(cur[1], 1),
                layers.shears[i];
                T = eltype(cur[1]), nScales = -1)
            @debug "size of daughters = $(size(daughters))"
            concatStart = reduce(+,
                q[1:i-1] .* resultingSize[1:i-1],
                init = 0) + 1
            # create an array for the fftPlan
            padBy = (0, 0)
        else
            concatStart = sum([prod(oneLayer[1:3]) for oneLayer in
                               outputSizes[1:i-1]]) + 1
            daughters = []
            padBy = getPadBy(layers.shears[i])
        end

        listOfProcessResults = Array{Future,1}(undef,
            size(cur[1])[dataDim+1] * length(cur))
        @debug "size of list = $(size(listOfProcessResults)), because " *
               "$(size(cur[1])[dataDim + 1]) and $(length(cur))"
        # parallel processing tools; the results are spread over the previous 2
        # layers paths

        # iterate over the scales two layers back
        if i == depth(layers)
            # compute the final mother if we're on the last layer
            if dataDim == 1
                @debug "last layer dataSizes[end] = $(dataSizes[end]), $(layers.shears[i + 1])"
                finalDaughters, ω = computeWavelets(dataSizes[end][1],
                    layers.shears[i+1]; nScales = 1)
                @debug "size of finalDaughters = $(size(finalDaughters))"
                concatStartLast = reduce(+, (q.*resultingSize)[1:end-1], init = 0) + 1
                @debug "concatStartLast = $(concatStartLast)"
                padByLast = (1, 3)
                nScalesLastLayer = size(daughters, 2) - 1
            else
                mother = []
                finalDaughters = []
                concatStartLast = sum([prod(oneLayer[1:3]) for oneLayer
                                       in
                                       outputSizes[1:end-1]]) + 1
                padByLast = getPadBy(layers.shears[end])
                nScalesLastLayer = q[end]
            end
            spawningJobs!(listOfProcessResults, layers, results, cur,
                outerAxes, innerAxes, innerSub, dataDim, i,
                daughters, finalDaughters, totalScales[i], nonlinear,
                subsam, outputSubsample, outputSizes, resultingSize,
                concatStart, concatStartLast, padBy, padByLast,
                nScalesLastLayer, fftPlans, T, dataSizes)
        else
            # any layer before the penultimate
            spawningJobs!(listOfProcessResults, layers, results, cur,
                dataSizes, outerAxes, innerAxes, innerSub, dataDim, i,
                daughters, totalScales[i], nonlinear, subsam,
                outputSubsample, outputSizes[i], concatStart, padBy,
                resultingSize, dataChannel, fftPlans, T)
        end
        # using one channel per
        for (λ, x) in enumerate(listOfProcessResults)
            @debug "type of the thing being fetched: $typeof(x)"
            tmpFetch = fetch(x)
            if typeof(tmpFetch) <: Exception
                throw(tmpFetch)
            end
            if i < depth(layers)
                nextData[λ] = tmpFetch
            end
        end
        @debug "fetched from workers"
    end
    results
end

# the midlayer version, i.e. i≠m
function spawningJobs!(listOfProcessResults, layers, results, cur, dataSizes,
    outerAxes, innerAxes, innerSub, dataDim, i, daughters,
    totalScale, nonlinear, subsam, outputSubsample,
    outputSize, concatStart, padBy, resultingSize,
    dataChannel, fftPlans, T)
    njobs = Distributed.nworkers()
    cjob = 0
    for (j, x) in enumerate(cur)
        # iterate over all paths in layer i
        @debug "siz =$(size(x, dataDim + 1))"
        for λ = 1:size(x, dataDim + 1)
            # do the actual transformation; the last layer requires
            # far less saved than the mid-layers. cjob+1 is the worker to
            # assign the task to (+1 since mod is 0...n-1 rather than 1...n
            @debug "sending out remotecalls"
            futureThing = remotecall(midLayer!, cjob + 1, layers, results, x,
                dataSizes, outerAxes, innerAxes, innerSub,
                dataDim, i, daughters, totalScale,
                nonlinear, subsam, outputSubsample,
                outputSize, concatStart, padBy, λ,
                resultingSize, dataChannel,
                fftPlans[cjob+1, i], T)
            cjob = mod(cjob + 1, njobs)
            if typeof(futureThing) <: typeof(Distributed.RemoteException)
                throw(futureThing)
            else
                @debug "writing to $((j - 1) * size(cur[1])[end] + λ) in listofProcessResults"
                listOfProcessResults[(j-1)*size(cur[1])[dataDim+1]+λ] = futureThing
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
    nScalesLastLayer, fftPlans, T, dataSizes)
    njobs = Distributed.nworkers()
    cjob = 0
    for (j, x) in enumerate(cur)
        # iterate over all paths in layer i
        for λ = 1:size(x, dataDim + 1)
            # do the actual transformation; the last layer requires
            # far less saved than the mid-layers

            futureThing = remotecall(finalLayer!, cjob + 1, layers, results, x,
                outerAxes, innerAxes, innerSub, dataSizes,
                dataDim, i, daughters, finalDaughters,
                totalScale, nonlinear, subsam,
                outputSubsample, outputSizes, λ,
                resultingSize, concatStart,
                concatStartLast, padBy, padByLast,
                nScalesLastLayer, fftPlans[cjob+1,
                    i], fftPlans[cjob+1, i+1], T)

            cjob = mod(cjob + 1, njobs)
            if typeof(futureThing) <: typeof(Distributed.RemoteException)
                throw(futureThing)
            else
                listOfProcessResults[(j-1)*size(cur[1])[end]+λ] = futureThing
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
function midLayer! end


viableDims = [(1, cwt)]
function midLayer!(layers::stParallel{K,1}, results, curPath, dataSizes,
    outerAxes, innerAxes, innerSub, dataDim, i, daughters,
    nScales, nonlinear, subsam, outputSubsample,
    outputSizeThisLayer, concatStart, padBy, λ, resultingSize,
    homeChannel, fftPlan, T) where {K}
    # make an array to return the results
    @debug "dataSizes = $dataSizes"
    toBeHandedBack = zeros(T, dataSizes[i+1][1:dataDim]...,
        size(daughters, 2) - 1,
        dataSizes[i+1][dataDim+2:end]...)
    @debug "size(toBeHandedBack) = $(size(toBeHandedBack))"
    # toBeHandedBack = SharedArray{T}(dataSizes[i+1][1:end-1]...,
    #                                    size(daughters,2)-1; pids=procs())
    # first perform the continuous wavelet transform on the data from the
    # previous layer
    output = cwt(view(curPath, innerAxes..., λ, outerAxes...), layers.shears[i],
        daughters, fftPlan)
    @debug "" size(output)
    innerMostAxes = axes(output)[end:end]
    # iterate over the non transformed dimensions of output
    writeLoop!(output, outerAxes, i, T, dataSizes, innerAxes, innerSub, layers,
        results, toBeHandedBack, concatStart, λ, dataDim, nothing,
        nonlinear, subsam, outputSizeThisLayer, outputSubsample,
        resultingSize)
    @debug "AFTER writeloop in midlayer size(toBeHandedBack) = $(size(toBeHandedBack))"
    return toBeHandedBack
end

function writeLoop!(output, outerAxes, i, T, dataSizes, innerAxes,
    innerSub, layers, results, toBeHandedBack,
    concatStart, λ, dataDim, outputSize,
    nonlinear, subsam, outputSizeThisLayer,
    outputSubsample, resultingSize)
    for outer in eachindex(view(output, [1 for i = 1:(dataDim+1)]...,
        outerAxes...))
        # output may have length zero if the starting path has low
        # enough scale
        for j = 2:size(output, dataDim + 1)
            toBeHandedBack[innerSub..., j-1, outer] =
                nonlinear.(resample(view(output, innerAxes..., j,
                        outer), layers.subsampling[i]))
        end
        # actually write to the output
        if subsam
            tmpOut = Array{T,dataDim}(real.(resample(view(output,
                    innerAxes...,
                    1,
                    outer),
                0.0f0, newSize = outputSizeThisLayer[1:dataDim])))
        else
            tmpOut = Array{T,dataDim}(real(resample(view(output,
                    innerAxes...,
                    1,
                    outer),
                layers.subsampling[i])))
        end
        sizeTmpOut = prod(size(tmpOut))
        if outputSubsample[1] > 1 || outputSubsample[2] > 1
            @debug "(λ-1)*resultingSize[i] = $((λ - 1) * resultingSize[i])"
            resampled = resample(tmpOut, 0.0f0, newSize = resultingSize[i])
            results[concatStart.+(λ-1)*resultingSize[i].+(0:(resultingSize[i]-1)), outer] =
                resample(tmpOut, 0.0f0, newSize = resultingSize[i])
        else
            results[concatStart.+(λ-1)*resultingSize[i].+(0:sizeTmpOut-1), outer] = reshape(tmpOut, (sizeTmpOut))
        end
        @debug "λ = $(λ), i=$(i), concatStart = $(concatStart)"
    end
end

@doc """
        finalLayer!(layers, results, curPath, outerAxes, innerAxes, innerSub, dataDim, i, daughters, nScales, nonlinear, subsam, outputSubsample, λ, resultingSize, concatStart, λ)
Do the transformation of the m-1 to mth layer, the output of the m-1st layer, and the output of the mth layer. These are glued together for mostly for computational reasons. There is currently no decreasing paths version, so it should end up throwing errors
"""
function finalLayer!(layers::stParallel{K,N}, results, curPath,
    outerAxes, innerAxes, innerSub, dataSizes, dataDim, i,
    daughters, finalDaughters, nScales, nonlinear,
    subsam, outputSubsample, outputSizes, λ, resultingSize,
    concatStart, concatStartLast, padBy, padByLast,
    nScalesLastLayer, fftPlan, fftPlanFinal, T) where {K,N}
    # iterate over the outerAxes
    # first perform the continuous wavelet transform on the data from the
    # previous layer
    @debug "inner = $(innerAxes), outer = $(outerAxes), size(curPath) = " *
           "$(size(curPath))"
    @debug "starting,λ=$(λ) in finalLayer!"
    output = dispatch(N, view(curPath, innerAxes..., λ, outerAxes...),
        layers.shears[i], daughters, fftPlan)

    @debug "starting, $innerAxes"
    nScalesPenult = getLastScales(dataDim, daughters, layers.shears[i])# numScales(layers.shears[i], size(innerAxes[1],1), i)
    toBeHandedOnwards = zeros(T, dataSizes[end][1:dataDim]...,
        nScalesPenult, dataSizes[end][3:end]...)
    @debug "handing onwards"
    writeLoop!(output, outerAxes, i, T, dataSizes, innerAxes, innerSub, layers,
        results, toBeHandedOnwards, concatStart, λ, dataDim, nothing,
        nonlinear, subsam, outputSizes[end-1], outputSubsample,
        resultingSize)
    @debug "writeLoop"
    innerAxes = axes(toBeHandedOnwards)[1:dataDim]
    # write the output from the mth layer to output
    tmpSize = 1
    for j = 2:size(output, dataDim + 1)
        finalOutput = averaging(dataDim, view(toBeHandedOnwards, innerAxes..., j - 1,
                outerAxes...), layers.shears[i+1],
            finalDaughters, fftPlanFinal)
        for outer in eachindex(view(finalOutput, [1 for i = 1:dataDim]...,
            outerAxes...))
            tmpFinal = Array{T,dataDim}(real.(resample(finalOutput[innerAxes...,
                    outer],
                0.0f0, newSize = outputSizes[end][1:dataDim])))
            if outputSubsample[1] > 1 || outputSubsample[2] > 1
                results[concatStartLast+
                        (λ-1)*nScalesLastLayer*resultingSize[end]+
                        (j-2)*resultingSize[i+1].+(0:(resultingSize[i+1]-1)),
                    outer] = resample(tmpFinal, 0.0f0, newSize = resampleTo(dataDim, resultingSize,
                    outputSizes, i))
            else
                sizeTmpOut = prod(size(tmpFinal))
                tmpSize = sizeTmpOut
                results[(concatStartLast+
                         (λ-1)*nScalesLastLayer*resultingSize[end]+
                         (j-2)*resultingSize[i+1]).+(0:(sizeTmpOut-1)),
                    outer] = reshape(tmpFinal, (sizeTmpOut))
            end
        end
    end
end

resampleTo(dataDim, resultingSize, outputSizes, i) = dataDim == 1 ?
                                                     resultingSize[i+1] : outputSizes[i+1][2]
