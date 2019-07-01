abstract type stType end
struct fullType <: stType end
struct decreasingType <: stType end


"""
    resized = getResizingRates(shears::Array{Shearlab.Shearletsystem2D{T}, 1}, M::Integer; percentage::Real=.9) where T<:Real
    getResizingRates(shears::layeredTransform; percentage::Real=.9)
2×(length of x) array of sizes the output should be to retain percentage of the mass of the averaging function.
  """
function getResizingRates(shears::Array{Shearlab.Shearletsystem2D{T}, 1},
                          M::Int; percentage=.9) where {T <: Real}
    newRates =zeros(Int64,M+1,2)
    for m=1:M+1
        tmpAveraging = abs.(ifftshift(shears[m].shearlets[:,:,end]))
        # since it's a product, we want to find when they're separately 90% their
        # total masses
        tmpAveragingx = tmpAveraging[:,1]/sum(tmpAveraging[:,1])
        tmpAveragingy = tmpAveraging[1,:]./sum(tmpAveraging[1,:])
        tmpSum = 0.0
        for i=1:length(tmpAveragingx)
            tmpSum += tmpAveragingx[i]
            if tmpSum>percentage/2
                newRates[m, 1] = i
                break
            end
        end
        tmpSum=0.0
        for j=1:length(tmpAveragingy)
            tmpSum+=tmpAveragingy[j]
            if tmpSum>percentage/2
                newRates[m, 2] = j
                break
            end
        end
    end
    2 .* newRates
end

getResizingRates(shears::layeredTransform{T}; percentage::Real=.9) where {T<:Real} = getResizingRates(shears.shears,shears.m; percentage=percentage)








@doc """
      n, q, dataSizes, outputSizes, resultingSize = calculateSizes(layers::layeredTransform{K,1}, outputSubsample, Xsize; totalScales = [-1 for i=1:layers.m+1]) where {K}

  * n is a list of the sizes of a single example. It is (m+1)×dataDim
one dimension at a time
  * q is a list of the number of scales, as used in 1D
  * datasizes is a list of tuples of sizes encountered in each layer
  * outputsizes is the same for the output arrays
  * resultingsize is a list of the output size as a result of subsampling
"""
function calculateSizes(layers::layeredTransform{K,1},
                        outputSubsample, Xsize; totalScales = [-1
                                                               for
                                                               i=1:layers.m+1],
                        percentage = .9) where {K}
    n = Int.(sizes(bsplineType(), layers.subsampling, Xsize[(end):end]))
    q = getQ(layers, n, totalScales)
    dataSizes = [Int.([Xsize[1:end-1]..., n[i], q[i]]) for i=1:layers.m+1]
    outputSizes = [Int.([Xsize[1:(end-1)]..., n[i+1], q[i]]) for i=1:layers.m+1]
    if outputSubsample[1] > 1
        resultingSize = zeros(Int, layers.m+1)
        resultingSubsampling = zeros(layers.m+1)
        # subsampling limited by the second entry
        for (i,x) in enumerate(outputSizes)
            proposedLevel = floor(Int, x[end-1]/outputSubsample[1])
            if proposedLevel < outputSubsample[2]
                # the result is smaller than the floor
                proposedLevel = outputSubsample[2]
            end
            resultingSize[i] = Int(proposedLevel)
        end
    elseif outputSubsample[2] > 1
        resultingSize = outputSubsample[2]*ones(Int64,layers.m+1)
    else
        resultingSize = [x[end-1] for x in outputSizes]
    end
    return (n, q, dataSizes, outputSizes, resultingSize)
end

"""

get the sizes of a single example in each layer
"""
function getNs(Xsize, layers::layeredTransform{K,1}) where K
    return Int.(sizes(bsplineType(), layers.subsampling, Xsize[(end):end]))
end

function calculateSizes(layers::layeredTransform{K,2},
                        outputSubsample, Xsize; totalScales=[-1 for
                                                             i=1:layers.m+1],
                        subsam=true, percentage = .9) where {K}
    dataXSizes = sizes(bsplineType(), layers.subsampling,
                       Xsize[(end-1):(end-1)])
    dataYSizes = sizes(bsplineType(), layers.subsampling, Xsize[(end):(end)])
    n = [reshape(dataXSizes, (layers.m+2, 1)) reshape(dataYSizes, (layers.m+2,
                                                                   1))]
    if subsam == true
        XOutputSizes = layers.outputSize[:, 1]
        YOutputSizes = layers.outputSize[:, 2]
    else
        XOutputSizes = dataXSizes
        YOutputSizes = dataYSizes
    end
    outputSizes = [(Int.(Xsize[1:end-2])..., Int.(XOutputSizes[m])...,
                    Int.(YOutputSizes[m])..., Int.(numInLayer(m-1,
                                                              layers)))
                   for m=1:layers.m+1]
    resultingSize = [XOutputSizes[m] *YOutputSizes[m] for m = 1:layers.m+1]
    q = getQ(layers, totalScales, product = false) .- 1
    dataSizes = [Int.([Xsize[1:end-2]..., dataXSizes[i],
                       dataYSizes[i], q[i]]) for i=1:layers.m+1]
    return (n, q, dataSizes, outputSizes, resultingSize)
end




@doc """
  q = getQ(layers, n, totalScales; product=true)
calculate the total number of entries in each layer
"""
function getQ(layers::layeredTransform{K,1}, n, totalScales; product=true) where {K}
    q = [numScales(layers.shears[i],n[i]) for i=1:layers.m+1]
    q = [(isnan(totalScales[i]) || totalScales[i]<=0) ? q[i] : totalScales[i] for i=1:layers.m+1]
    if product
        q = [prod(q[1:i-1]) for i=1:layers.m+1]
        return q
    else
        return q
    end
end
@doc """
  q = getQ(layers, n, totalScales; product=true)
calculate the total number of shearlets/wavelets in each layer
"""
function getQ(layers::layeredTransform{K, 2}, totalScales; product=true) where {K}
    q = [numScales(layers.shears[i]) for i=1:layers.m+1]
    q = [(isnan(totalScales[i]) || totalScales[i]<=0) ? q[i] : totalScales[i] for i=1:layers.m+1]
    if product
        q = [prod(q[1:i-1]) for i=1:layers.m+1]
        return q
    else
        return q
    end
end

getQ(layers, Xsize; product=true) = getQ(layers, Int.(sizes(bspline, layers.subsampling, Xsize[(end):end]))
, [-1 for i=1:layers.m+1]; product=product)








function getPadBy(shearletSystem::Shearlab.Shearletsystem2D{T}) where T
    padBy = (0,0)
    for sysPad in shearletSystem.support
        padBy = (max(sysPad[1][2]-sysPad[1][1], padBy[1]), max(padBy[2], sysPad[2][2]-sysPad[2][1]))
    end
    return padBy
end












import LinearAlgebra.norm
function norm(scattered::scattered{T,N}, p) where {T<:Number, N}
    #TODO functional version of this
    # entrywise = zeros(T,size(scattered.output)[1:(end-2)])
    # innerAxes = axes(scattered.output)[end-1:end]
    # for outer in eachindex(view(scatte))
    return (sum([norm(scattered.output[i],p).^p for i=1:scattered.m+1])).^(1/p)
end


function pad(x, padBy)
    T = eltype(x)
    firstRow = cat(zeros(T, size(x)[1:end-2]..., padBy...),
                   zeros(T, size(x)[1:end-2]..., padBy[1], size(x)[end]),
                   zeros(T, size(x)[1:end-2]..., padBy...), dims = length(size(x)))
    secondRow = cat(zeros(T, size(x)[1:end-1]..., padBy[2]), x,
                    zeros(T, size(x)[1:end-1]..., padBy[2]), dims=length(size(x)))
    thirdRow = cat(zeros(T, size(x)[1:end-2]..., padBy...), zeros(T, size(x)[1:end-2]...,
                                                              padBy[1], size(x)[end]),
                   zeros(T, size(x)[1:end-2]..., padBy...), dims = length(size(x)))
    return cat(firstRow, secondRow, thirdRow, dims = length(size(x))-1)
end



# TODO maybe fold this in?
"""
    totalSize, Xsizes, Ysizes = outputSize(X,layers)

get the length of a thin version
"""
function outputSize(X, layers)
  Xsizes = [size(X,1); layers.reSizingRates[1,:]]
  Ysizes = [size(X,2); layers.reSizingRates[2,:]]
  return (sum([numInLayer(m-1,layers)*Xsizes[m+1]*Ysizes[m+1] for m=1:layers.m+1]), Xsizes, Ysizes)
end



"""
    createFFTPlans(layers::layeredTransform{K, 2}, dataSizes; verbose=false, T=Float32, iscomplex::Bool=false)

to be called after distributed. returns a 2D array of futures; the i,jth entry has a future (referenced using fetch) for the fft to be used by worker i in layer j.
"""
function createFFTPlans(layers::layeredTransform, dataSizes; verbose=false,
                        T=Float32, iscomplex::Bool=false)
    FFTs = Array{Future,2}(undef, nworkers(), layers.m+1)
    for i=1:nworkers(), j=1:layers.m+1
        if verbose
            println("i=$(i), j=$(j)")
        end
        if length(layers.n) >= 2
            padBy = getPadBy(layers.shears[j])
            FFTs[i, j] = remotecall(createRemoteFFTPlan, i,
                                    dataSizes[j][(end-2):(end-1)], padBy, T,
                                    iscomplex)
        else
            padBy = (0, 0)
            fftPlanSize = (dataSizes[j][1:(end-2)]..., 2*dataSizes[j][end-1])

            FFTs[i, j] = remotecall(createRemoteFFTPlan, i, fftPlanSize,
                                    T, iscomplex)
        end
    end
    return FFTs
end


"""
the 1D version
"""
function createRemoteFFTPlan(sizeOfArray, T, iscomplex)
    if iscomplex
        return plan_fft!(zeros(T, sizeOfArray...), length(sizeOfArray), flags =
                         FFTW.PATIENT)
    else
        return plan_rfft(zeros(T, sizeOfArray...), length(sizeOfArray), flags =
                         FFTW.PATIENT)
    end
end
"""
the 2D version
"""

function createRemoteFFTPlan(sizeOfArray, padBy, T, iscomplex)
    if iscomplex
        return plan_fft!(pad(zeros(T, sizeOfArray...), padBy), flags = FFTW.PATIENT)
    else
        return plan_rfft(pad(zeros(T, sizeOfArray...), padBy), flags = FFTW.PATIENT)
    end
end

function remoteMult(x::Future,y)
    fetch(x)*y
end
remoteMult(y,x::Future) = remoteMultiply(x,y)

function remoteDiv(x::Future,y)
    fetch(x) / y
end
remoteDiv(y,x::Future) = remoteMultiply(x,y)
