# Actual scattering function
########################################################
########################################################
# TODO: order shouldn't matter since the types are unique
@doc """
  output = st(X::Array{Float64,1}, layers::layeredTransform; nonlinear::Function=abs, subsam::Function=subsample, stType::String, collator::Array{Array{float64},1}=Array{Array{float64},1}(0))

  1D scattering transform using the layeredTransform layers. you can switch out the nonlinearity as well as the method of subsampling. Finally, the stType is a string. If it is "full", it will produce all paths. If it is "decreasing", it will only keep paths of increasing scale. If it is "collating", you must also include a vector of matrices.
"""
function st(X::Array{T, N}, layers::layeredTransform,
            nonlinear::nl, fullOr::fullType=fullType();
            thin::Bool=false, subsam::Bool=true,
            totalScales = [-1 for i=1:layers.m+1]) where {T<:Real,
                                                          S<:Union, N, nl <:
                                                          nonlinearity}
    @assert length(totalScales)==layers.m+1
    # Insist that X has to have at least one extra meta-dimension, even if it is 1
    if length(size(X)) == elTypes(layers)[2]
        X = reshape(X, (1,size(X)...));
    end
    numChildλ = 0
    dataDim, n, q, dataSizes, outputSizes, resultingSize =
        calculateSizes(layers, (-1,-1), size(X), totalScales = totalScales)
    results = scattered(layers, X, stType; totalScales=totalScales)
    nScalesLayers = [numScales(layers.shears[i], max(dataSizes[i][1]-1,1)) for
                     i=1:length(layers.shears)] 
    for (i,layer) in enumerate(layers.shears)
        cur = results.data[i] #data from the previous layer
        innerAxes = axes(cur)[end-dataDim:end-1] # effectively a set of colons
        # of length dataDim, to be used for input
        innerSub = (Base.OneTo(x) for x in outputSizes[i][end-dataDim:end-1]) # effectively a set of colons of length dataDim, to be used for the subsampled output
        
        P = plan_rfft(results.data, flags=FFTW.PATIENT)
        outerAxes = axes(cur)[1:(end-results.dataDim-1)] # the same idea as
        # innerAxes, but for the example indices

        # precompute the wavelets
        if stType=="decreasing" && i>=2
            numChildλ = numChildren(keeper, layers, nScalesLayers)
            daughters = computeWavelets(size(cur)[end-1], layers.shears[i], nScales=numChildλ)
        elseif i<layers.m+1
            daughters = computeWavelets(size(cur)[end-1], layers.shears[i],nScales=totalScales[i])
        elseif i==layers.m+1
            daughters = computeWavelets(size(cur)[end-1], layers.shears[i],nScales=1)
        end
        if i<=layers.m
            decreasingIndex = 1
            for λ = 1:size(cur)[end]
                # first perform the continuous wavelet transform on the data from the previous layer
                if stType=="decreasing" && i>=2
                    numChildλ = numChildren(keeper, layers, nScalesLayers)
                    output = cwt(cur[outerAxes, innerAxes..., λ], layers.shears[i], daughters, J1=numChildλ - 1 + layers.shears[i].averagingLength)
                    if size(output)[end] != numChildλ+1
                        error("size(output,2)!=numChildλ $(size(output,2))!=$(numChildλ+1)")
                    end
                elseif stType=="decreasing"
                    output = cwt(cur[outerAxes..., innerAxes..., λ], layers.shears[i], daughters)
                else
                    output = cwt(cur[outerAxes..., innerAxes..., λ], layers.shears[i], daughters)
                end
                # TODO: output may have length zero if the starting path has low enough scale
                # iterate over the outerAxes
                for outer in eachindex(view(cur, outerAxes..., [1 for i=1:results.k+1]...))
                    for j = 2:size(output)[end]
                        if stType == "full"
                            # println("$outer, $innerSub, $(λ-1)*$((size(output)[end]-1))+$(j-1)")
                            results.data[i+1][outer, innerSub..., (λ-1)*(size(output)[end]-1)+j-1] = nonlinear.(subsam(output[outer, innerAxes..., j], layers.subsampling[i]))
                        else
                            if stType=="decreasing"
                                results.data[i+1][outer, innerSub..., decreasingIndex] = nonlinear.(subsam(output[outer, innerAxes..., j], layers.subsampling[i]))
                                decreasingIndex += 1
                            end
                        end
                    end
                    # println("THING BROKE")
                    # println("results.output[i][outer, innerSub..., λ] =$(size(results.output[i][outer, innerSub..., λ]))")
                    # println("size(output) = $(size(output))")
                    # println("innerAxes = $(innerAxes)")
                    # println("size(output[outer, innerAxes..., end]) = $(size(output[outer, innerAxes..., size(output)[end]]))")
                    # println("nonlinear.(real(subsam(output[outer, innerAxes..., end], layers.subsampling[i]))) $(nonlinear.(real(subsam(output[outer, innerAxes..., size(output)[end]], layers.subsampling[i]))))")
                    results.output[i][outer, innerSub..., λ] = Array{Float64,results.k}(nonlinear.(real(subsam(output[outer, innerAxes..., size(output)[end]], layers.subsampling[i]))))
                end
                if stType=="decreasing" && i>1
                    isEnd, keeper = incrementKeeper(keeper, i-1, layers, nScalesLayers)
                end
            end
        else
            for λ = 1:size(cur)[end]
                output = cwt(cur[outerAxes..., innerAxes..., λ], layers.shears[i], daughters, nScales=1)
                for outer in eachindex(view(cur, outerAxes..., [1 for i=1:results.k]..., 1))
                    results.output[i][outer, innerSub..., λ] = Array{Float64,results.k}(nonlinear.(real(subsam(output[outer, innerAxes..., size(output)[end]], layers.subsampling[i]))))
                end
            end
        end
    end
return results
end







st(layers::layeredTransform, X::Array{T}; nonlinear::Function=abs, subsam::Function=bspline) where {T<:Real} = st(X,layers,nonlinear=nonlinear, subsamp=subsam)




# TODO: make a function which determines the breakpoints given the layers function and the size of the input
@doc """
        thinSt(X::Array{T}, layers::layeredTransform; nonlinear::Function=abs, subsam::Function=bspline, stType::String="full", outputSubsample::Tuple{Int,Int}=(-1,1), totalScales = [-1 for i=1:layers.m+1]) where {T<:Real}

      See the main st for a description of the options. This is a version of the 1D st that only returns a concatinated vector for output. It is most useful for classification applications.

    outputSubsample is a tuple, with the first number indicating a rate, and the second number giving the minimum allowed size. If some of the entries are less than 1, it has different behaviour:
    (<1, x): subsample to x elements for each path.
    (<1, <1): no ssubsampling
    (x, <1) subsample at a rate of x, with at least one element kept in each path

    totalScales, if positive, gives the number of non-averaging wavelets.
TODO: non-thin is currently broken
    """
function thinSt(X::Array{T, N}, layers::layeredTransform, nonlinear::nl,
                fullOr::fullType=fullType();# subsam::Sub = bspline(), #TODO
                # allow for variation in subsamping method again
                thin::Bool=true, outputSubsample::Tuple{Int, Int}=(-1,-1),
                subsam::Bool=true, totalScales = [-1 for i=1:layers.m+1]) where {T<:Real,
                                                                                 S<:Union, N, nl <: nonlinearity, Sub <: resamplingMethod}
    @assert length(totalScales)==layers.m+1
    dataDim = elTypes(layers)[2]
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
                                                                 totalScales)
    if thin
        nextData = [reshape(X, (size(X)..., 1)),]
        # get the size of a single entry as a long string
        if dataDim==1 && outputSubsample[1] > 1 || outputSubsample[2] > 1
            netSize = Int(sum(resultingSize.*q))
        else
            netSize = Int(sum([prod(x[end-dataDim:end]) for x in outputSizes])...)
        end
        # create list of size references if we're subsampling the output an
        # extra amount
        results = SharedArray(zeros(T, outputSizes[1][1:end-dataDim-1]...,
                                    Int(sum(resultingSize .* q))))
    else
        error("the non-thin st is currently in the works");
        results = scattered(layers, X, totalScales = totalScales)
    end

    nScalesLayers = [numScales(layers.shears[i],
                               max(dataSizes[i][1]-1,1)) for
                     i=1:length(layers.shears)]
    iterateOverLayers()
    return results
end


function iterateOverLayers(layers, results, input, dataSizes, outputSizes,
                           dataDim, q, totalScales, T)
    for (i,layer) in enumerate(layers.shears[1:layers.m])
        cur = nextData #data from the previous layer
        # only store the intermediate results in intermediate layers
        if i < layers.m
            # store the data in a channel, along with the number of
            # it's parent
            dataChannel = Channel{ Tuple{ Array{ length(dataSizes[i+1])},
                                          Int64}}(size(cur)[end])
            nextData = Array{SharedArray{Float64, length(dataSizes[i+1])},
                             1}(undef, size(cur[1])[end]*length(cur))
        end

        innerAxes = axes(cur[1])[end-dataDim:end-1] # effectively a
        # set of colons of length dataDim, to be used for input

        innerSub = (Base.OneTo(x) for x in
                    outputSizes[i][end-dataDim:end-1]) # effectively a
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
        else
            concatStart = reduce(+,
               q[1:i-1] .* [resultingSize[j][1] for j =
                            1:i-1] .* [resultingSize[j][2] for j = 1:i-1],
               init = 0) + 1
            daughters = []
            padBy = getPadBy(layers.shears[i])
        end
        if T<:Real
            fftPlan = plan_rfft(cur[(end-dataDim+1):end], flags = FFTW.PATIENT)
        elseif T<:Complex
            fftPlan = plan_fft(cur[(end-dataDim+1):end], flags = FFTW.PATIENT)
        else
            error("we have data in an array that is neither" *
                  " real nor complex")
        end
        listOfProcessResults = Array{Future, 1}(undef,
                                                size(cur[1])[end]*length(cur))
        # parallel processing tools; the results are spread over the previous 2
        # layers paths

        # iterate over the scales two layers back
        if i==layers.m
            # compute the final mother if we're on the last layer
            if dataDim==1
                mother = computeWavelets(size(cur[1])[end-1],
                                         layers.shears[i+1], nScales=1)
                concatStartLast = sum(q[1:end-1] .*
                    resultingSize[1:end-1]) + 1
                padByLast = getPadBy(layers.shears[end])
            else
                mother = []
                concatStartLast = sum(q[1:end-1] .*
                    outputSize[1:end-1][1] .* outputSize[1:end-1][2]) + 1
                padByLast = getPadBy(layers.shears[end])
            end
            # make a final plan
            if T<:Real
                fftPlan = plan_rfft(cur[(end-dataDim+1):end], flags = FFTW.PATIENT)
            elseif T<:Complex
                fftPlan = plan_fft(cur[(end-dataDim+1):end], flags = FFTW.PATIENT)
            end
            
            nScalesLastLayer = getQ(layers, n, totalScales, product =
                false)[end-1]
            spawningJobs!(listOfProcessResults, layers, results, x, outerAxes,
                          innerAxes, innerSub, dataDim, i, daughters,
                          totalScales[i], nonlinear, subsam, outputSubsample,
                          λ, resultingSize, concatStart, concatStartLast,
                          padBy, padByLast, nScalesLastLayer, fftPlan, fftPlanFinal, T)
        else 
            spawningJobs!(listOfProcessResults, layers, results, x, dataSizes,
                          outerAxes,innerAxes, innerSub, dataDim, i, daughters,
                          totalScales[i], nonlinear, subsam, outputSubsample,
                          outputSizes[i], concatStart, padBy, λ, resultingSize,
                          dataChannel, fftPlan, T)
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
    end
    results
end

# the midlayer version, i.e. i≠m
function spawningJobs!(listOfProcessResults, layers, results, x, dataSizes,
                       outerAxes, innerAxes, innerSub, dataDim, i, daughters,
                       totalScale, nonlinear, subsam, outputSubsample,
                       outputSize, concatStart, padBy, λ, resultingSize,
                       dataChannel, fftPlan, T)
    for (j,x) in enumerate(cur)
        # iterate over all paths in layer i
        for λ = 1:size(x)[end]
            # do the actual transformation; the last layer requires
            # far less saved than the mid-layers
            futureThing = @spawn midLayer!(layers, results, x, dataSizes,
                                           outerAxes, innerAxes, innerSub,
                                           dataDim, i, daughters, totalScale,
                                           nonlinear, subsam, outputSubsample,
                                           outputSize, concatStart, padBy, λ,
                                           resultingSize, dataChannel, fftPlan,
                                           T)
            if typeof(futureThing) <: typeof(Distributed.RemoteException)
                throw(futureThing)
            else
                listOfProcessResults[(j-1)*size(cur[1])[end] + λ] = futureThing
            end
        end
    end
end

# the end version, i.e. i=m
function spawningJobs!(listOfProcessResults, layers, results, x, outerAxes,
                       innerAxes, innerSub, dataDim, i, daughters, totalScale,
                       nonlinear, subsam, outputSubsample, λ, resultingSize,
                       concatStart, concatStartLast, padBy, padByLast,
                       nScalesLastLayer, fftPlan, fftPlanLast, T)
    for (j,x) in enumerate(cur)
        # iterate over all paths in layer i
        for λ = 1:size(x)[end]
            # do the actual transformation; the last layer requires
            # far less saved than the mid-layers
            futureThing = @spawn finalLayer!(layers, results, x, outerAxes,
                                             innerAxes, innerSub, dataDim, i,
                                             daughters, totalScale, nonlinear,
                                             subsam, outputSubsample, λ,
                                             resultingSize, concatStart,
                                             concatStartLast, padBy, padByLast,
                                             nScalesLastLayer, fftPlan,
                                             fftPlanLast, T)
        end
        if typeof(futureThing)<: typeof(Distributed.RemoteException)
            throw(futureThing)
        else
            listOfProcessResults[(j-1)*size(cur[1])[end] + λ] = futureThing
        end
    end
    return listOfProcessResults
end

for nl in functionTypeTuple
    @eval begin

        @doc """
        midLayer!(layers, results, cur, dataSize, outerAxes, innerAxes, innerSub, dataDim, i, daughters, nScales, nonlinear, subsam, outputSubsample, concatStart, λ)
        midLayer!(layers, results, cur, nextData, outerAxes, innerAxes, innerSub, outerAxes, dataDim, i, nScalesLayers, daughters, nScales, nonlinear, subsam, outputSubsample,
     λ, keeper)
    An intermediate function which takes a single path in a layer and generates all of the children of that path in the next layer. Only to be used on intermediate layers, as finalLayerTransform fills the same role in the last layer. If keeper is omitted, it assumes we don't have decreasing type, otherwise it's assumed

    """
        function midLayer!(layers::layeredTransform{K,1}, results, curPath, dataSize,
                           outerAxes, innerAxes, innerSub, dataDim, i, daughters,
                           nScales, nonlinear::$(nl[2]), subsam, outputSubsample,
                           outputSizeThisLayer, concatStart, padBy, λ, resultingSize,
                           homeChannel, fftPlan, T) where {K}
            # make an array to return the results
            toBeHandedBack = zeros(T, outputSizeThisLayer..., size(daughters,2)-1)
            # first perform the continuous wavelet transform on the data from the previous layer
            output = cwt(curPath[outerAxes..., innerAxes..., λ], layers.shears[i],
                         daughters, nScales=nScales, fftPlan=fftPlan)
            innerMostAxes = axes(output)[end:end]
            # iterate over the non transformed dimensions of output
            for outer in eachindex(view(output, outerAxes..., [1 for i=1:(dataDim+1)]...))
                # output may have length zero if the starting path has low enough scale
                for j = 2:size(output)[end]
                    toBeHandedBack[outer, innerSub..., j-1] =
                        $(nl[1]).(resample(output[outer, innerAxes..., j],
                                           layers.subsampling[i]))
                end
                # actually write to the output
                tmpOut = Array{T, dataDim}(real(resample(output[outer,
                                                                innerAxes...,
                                                                end],
                                                         layers.subsampling[i])))
                sizeTmpOut = prod(size(tmpOut))
                if outputSubsample[1] > 1 || outputSubsample[2] > 1
                    results[outer, concatStart .+ (λ-1)*resultingSize[i] .+
                            (0:(resultingSize[i]-1))] = resample(tmpOut,
                                                                 resultingSize[i],
                                                                 absolute = true)
                else
                    results[outer, concatStart .+ (λ-1)*resultingSize[i] .+
                            (0:sizeTmpOut-1)] = reshape(tmpOut, (sizeTmpOut))
                end
            end
            return toBeHandedBack
        end


        function midLayer!(layers::layeredTransform{K,2}, results, curPath, dataSize,
                           outerAxes, innerAxes, innerSub, dataDim, i, daughters,
                           nScales, nonlinear::$(nl[2]), subsam, outputSubsample,
                           outputSizeThisLayer, concatStart, padBy, λ, resultingSize,
                           homeChannel, fftPlan, T) where {K}
            # make an array to return the results
            toBeHandedBack = zeros(T, outputSizeThisLayer..., size(daughters, 2)-1)
            # first perform the continuous wavelet transform on the data from the
            # previous layer
            output = sheardec2D(curPath[outerAxes..., innerAxes..., λ],
                                layers.shears[i], fftPlan, true, padBy)
            innerMostAxes = axes(output)[end:end]
            # iterate over the non transformed dimensions of output
            for outer in eachindex(view(output, outerAxes..., [1 for
                                                               i=1:(dataDim+1)]...))
                # output may have length zero if the starting path has low enough scale
                for j = 2:size(output)[end]
                    toBeHandedBack[outer, innerSub..., j-1] =
                        $(nl[1]).(resample(output[outer, innerAxes..., j],
                                           layers.subsampling[i]))
                end
                # actually write to the output
                if subsam
                    tmpOut = Array{T, dataDim}(real.(resample(output[outer,
                                                                     innerAxes...,
                                                                     end],
                                                              0f0, newsize =
                                                              resultingSize[i][end-2:end-1]))) 
                else 
                    tmpOut = Array{T, dataDim}(real.(resample(output[outer,
                                                                     innerAxes...,
                                                                     end],
                                                              layers.subsampling[i])))
                end
                sizeTmpOut = prod(size(tmpOut))
                results[outer, concatStart .+ (λ-1)*resultingSize[i] .+
                        (0:sizeTmpOut-1)] = reshape(tmpOut, (sizeTmpOut))
            end
            return toBeHandedBack
        end

@doc """
        finalLayer!(layers, results, curPath, outerAxes, innerAxes, innerSub, dataDim, i, daughters, nScales, nonlinear, subsam, outputSubsample, λ, resultingSize, concatStart, λ)

    Do the transformation of the m-1 to mth layer, the output of the m-1st layer, and the output of the mth layer. These are glued together for mostly for computational reasons. There is currently no decreasing paths version, so it should end up throwing errors
    """
function finalLayer!(layers::layeredTransform{K,1}, results, curPath,
                     outerAxes, innerAxes, innerSub, dataDim, i, daughters,
                     nScales, nonlinear::$(nl[2]), subsam, outputSubsample, λ,
                     resultingSize, concatStart, concatStartLast, padBy,
                     padByLast, nScalesLastLayer, fftPlan, fftPlanLast, T) where {K}
    localIndex = 0
    # iterate over the outerAxes
    # first perform the continuous wavelet transform on the data from the previous layer
    output = cwt(curPath[outerAxes..., innerAxes..., λ], layers.shears[i],
                 daughters, nScales=nScales, fftPlan = fftPlan)
    for outer in eachindex(view(output, outerAxes..., [1 for
                                                       i=1:(dataDim+1)]...))
        # write the output from the m-1st layer to output
        tmpOut = Array{T, dataDim}($(nl[1])(real(subsam(output[outer, innerAxes...,
                                                               end],
                                                        layers.subsampling[i]))))
        if outputSubsample[1] > 1 || outputSubsample[2] > 1
            results[outer, concatStart .+ (λ-1)*resultingSize[i] .+
                    (0:(resultingSize[i]-1))] = resample(tmpOut, resultingSize[i],
                                                         absolute = true)
        else
            sizeTmpOut = prod(size(tmpOut))
            results[outer, concatStart .+ (λ-1)*resultingSize[i].+(0:(sizeTmpOut-1))] = reshape(tmpOut, (sizeTmpOut))
        end
    end
    # write the output from the mth layer to output
    for j = 2:size(output)[end]
        finalOutput = cwt(output[outerAxes..., innerAxes..., λ],
                          layers.shears[i+1], daughters, fftPlan=fftPlanLast, nScales=1)
        for outer in eachindex(view(finalOutput, outerAxes..., [1 for
                                                                i=1:(dataDim+1)]...))
            tmpFinal = Array{Float64,dataDim}(real(subsam(finalOutput[outer,
                                                                      innerAxes...,
                                                                      end],
                                                          layers.subsampling[i]))) 
            if outputSubsample[1] > 1 || outputSubsample[2] > 1
                results[outer, concatStartLast +
                        (λ-1)*nScalesLastLayer*resultingSize[i] +
                        (j-2)*resultingSize[i] .+ (0:(resultingSize[i+1]-1))] =
                        subsam(tmpFinal, resultingSize[i+1], absolute = true)
            else
                sizeTmpOut = prod(size(tmpFinal))
                results[outer, (concatStartLast +
                                (λ-1)*nScalesLastLayer*resultingSize[i] +
                                localIndex +
                                (λ-1)*resultingSize[i+1]).+(0:(sizeTmpOut-1))] = reshape(tmpFinal, (sizeTmpOut))
            end
        end
    end
end


@doc """
        finalLayer!(layers, results, curPath, outerAxes, innerAxes, innerSub, dataDim, i, daughters, nScales, nonlinear, subsam, outputSubsample, λ, resultingSize, concatStart, λ)

    Do the transformation of the m-1 to mth layer, the output of the m-1st layer, and the output of the mth layer. These are glued together for mostly for computational reasons. There is currently no decreasing paths version, so it should end up throwing errors
    """
function finalLayer!(layers::layeredTransform{K, 2}, results, curPath,
                     outerAxes, innerAxes, innerSub, dataDim, i, daughters,
                     nScales, nonlinear::$(nl[2]), subsam, outputSubsample, λ,
                     resultingSize, concatStart, concatStartLast,
                     nScalesLastLayer, fftPlan, T) where {K} 
    localIndex = 0
    # iterate over the outerAxes
    # first perform the continuous wavelet transform on the data from the previous layer
    output = sheardec2D(curPath[outerAxes..., innerAxes..., λ],
                        layers.shears[i], fftPlan, true, padBy)
    for outer in eachindex(view(output, outerAxes..., [1 for
                                                       i=1:(dataDim+1)]...))
        # write the output from the m-1st layer to output
        tmpOut = Array{T, dataDim}(real(subsam(output[outer, innerAxes...,
                                                      end],
                                               layers.subsampling[i])))
        if outputSubsample[1] > 1 || outputSubsample[2] > 1
            results[outer, concatStart .+ (λ-1)*resultingSize[i] .+
                    (0:(resultingSize[i]-1))] = resample(tmpOut, resultingSize[i],
                                                         absolute = true)
        else
            sizeTmpOut = prod(size(tmpOut))
            results[outer, concatStart .+
                    (λ-1)*resultingSize[i].+(0:(sizeTmpOut-1))] = reshape(tmpOut,
                                                                          (sizeTmpOut))
        end
    end
    # write the output from the mth layer to output
    for j = 2:size(output)[end]
        finalOutput = averagingFunction(curPath[outerAxes..., innerAxes..., λ],
                                        layers.shears[i+1], fftPlanFinal)
        for outer in eachindex(view(finalOutput, outerAxes..., [1 for
                                                                i=1:(dataDim+1)]...))
            tmpFinal = Array{Float64,dataDim}(real(subsam(finalOutput[outer,
                                                                      innerAxes...,
                                                                      end],
                                                          layers.subsampling[i])))
            if outputSubsample[1] > 1 || outputSubsample[2] > 1
                results[outer, concatStartLast +
                        (λ-1)*nScalesLastLayer*resultingSize[i] +
                        (j-2)*resultingSize[i] .+ (0:(resultingSize[i+1]-1))] =
                        subsam(tmpFinal, resultingSize[i+1], absolute = true)
            else
                sizeTmpOut = prod(size(tmpFinal))
                results[outer, (concatStartLast +
                                (λ-1)*nScalesLastLayer*resultingSize[i] +
                                localIndex +
                                (λ-1)*resultingSize[i+1]).+(0:(sizeTmpOut-1))] = reshape(tmpFinal, (sizeTmpOut))
            end
        end
    end
end
end
end
# function midLayer!(layers, results, cur, nextData, outerAxes, innerAxes, innerSub, dataDim, i, nScalesLayers, daughters, J1, nonlinear, subsam, outputSubsample, λ, concatStart, keeper)
#    # TODO: This (the decreasing paths version) is utterly broken atm. Will return and fix it
#   decreasingIndex = 1
#   # iterate over the outerAxes
#   for outer in eachindex(view(cur, outerAxes..., [1 for i=1:dataDim]..., 1))
#     # first perform the continuous wavelet transform on the data from the previous layer
#     if i>=2
#       output=cwt(cur[outer, innerAxes...,λ], layers.shears[i], daughters, J1=J1)
#     else
#       output = cwt(cur[outer, innerAxes..., λ], layers.shears[i], daughters)
#     end
#     # output may have length zero if the starting path has low enough scale
#     for j = 2:size(output,2)
#       nextData[outer, innerSub..., decreasingIndex] = nonlinear.(subsam(output[innerAxes..., j], layers.subsampling[i]))
#       decreasingIndex += 1
#     end
#     if i>1
#       isEnd, keeper = incrementKeeper(keeper, i-1, layers, nScalesLayers)
#     end
#     tmpOut = Array{Float64,dataDim}(real(subsam(output[innerAxes..., end], layers.subsampling[i])))
#     sizeTmpOut = prod(size(tmpOut))
#     if outputSubsample[1] > 1 || outputSubsample[2] > 1
#       results[outer, concatStart .+ (0:(resultingSize[i]-1))] = subsam(tmpOut, resultingSize[i], absolute = true)
#       sizeTmpOut = resultingSize[i]
#     else
#       #TODO this is a sequential way of accessing the location
#       results[outer, concatStart.+(0:(sizeTmpOut-1))] = reshape(tmpOut, (sizeTmpOut))
#     end
#   end
# end
