using Shearlab
using Plots
using ScatteringTransform
using MLDatasets
pyplot()

# see the error in the bilinear re-interpolation
k = 1:1/27:4
f = [(t-3).^2 .* (s+2).*exp.(-t^2/50)+abs(t-s) for s=k,t=k]
heatmap(k,k,f)
fSamp = shatteringTransform.resample(f,2.0)
fReSamp = shatteringTransform.resample(fSamp,1/2.0)
heatmap(range(1,stop=4,length=size(fReSamp,1)),range(1,stop=4,length=size(fReSamp,2)), real(fReSamp))
errMatrix = zeros(28,28)
for i=1:size(fReSamp,1),j=1:size(fReSamp,2)
    if f[i,j]<1e-8
        errMatrix[i,j] = abs(real(fReSamp[i,j])-f[i,j])
    else
        errMatrix[i,j] = abs(real(fReSamp[i,j])-f[i,j])./abs(f[i,j])
    end
end
old = 49; neww= 9; tmp = [[x,y] for x=range(1,stop=old,range=neww),y=range(1,stop=old,range=neww)]; scatter([x[1] for x in tmp], [x[2] for x in tmp],label="")
heatmap(errMatrix)
maximum(errMatrix)
heatmap(real(fSamp))

# DELETE ABOVE HERE

train_x, train_y = MNIST.traindata();
n = 1234; example = Float32.(train_x[:, :, n]); train_y[n]
heatmap(example)
layers = layeredTransform(2, size(example), subsamples = [28/19, 19/13, 13/8])
@time fullySheared = st(example, layers, softplusType(), thin=false)
@time reconstructionSp, backShatter = pseudoInversion(fullySheared,layers,softplusType())
heatmap(flipdim(reconstructionSp,1))
heatmap(flipdim(example,1))

fullySheared.output
heatmap(real.(fullySheared.output[1][:,:,1]))


# create an array of the resampled scattering transform outputs
function pseudoInversion(fullySheared::shattered{T}, layers::layeredTransform, nonlin::S) where {T, S<:nonlinearity}
    # we're going to assume that fullySheared has nothing meaningful in its data portion

    zerothLayerUpsample = upSample(fullySheared.output[1][:,:,1],(28,28))
    fullySheared.output[1] = reshape(zerothLayerUpsample,(size(zerothLayerUpsample)...,1))
    # point of order: we need fft plans
    fftPlans = createFFTPlans(layers, [size(dat) for dat in fullySheared.data],
                              iscomplex= (T<:Complex))

    # start by upsampling all of the outputs and storing them in the data in
    # all but the last layer
    resampleLayers!(fullySheared.data, fullySheared.output, nonlin)

    # once that's done, we start at the deepest layers and work our way up. In
    # the deepest layer, we have a lossy estimate of the data, since we have no
    # information about the higher frequencies at this layer; additionally, we
    # have a final nonlinearity to undo
    estimateTheLastLayer!(fullySheared.output[end], layers.shears[end],
                          fullySheared.data[end], nonlin, fftPlans[end])

    for m=layers.m:-1:2
        estimateMidLayer!(fullySheared.data[m], fullySheared.output,
                          layers.shears[m], nonlin)
    end

    # we've worked backwards through the data; now reconstruct the input
    reconstruction =
        shearrec2D(cat(3,[upSample(aTanh.(backShatter.data[2][:,:,i]),size(backShatter.output[1][:,:,1]))
    for i=1:size(fullySheared.data[2],3)]...,
                   backShatter.output[1][:,:,1]),layers.shears[1])
    backShatter.data[1] = reshape(reconstruction,(size(reconstruction)...,1))
    return (reconstruction,backShatter)
end

function resampleLayers!(layerData, layerOutput, nonlin)
    for l = layers.m:-1:2
        innerAxis = axes(layerData[l])[(end-2):(end-1)]
        outerAxis = axes(layerData[l])[1:(end-2)]
        for outer in eachindex(view(layerOutput[l], outerAxis, 1, 1,1))
            for i= 1:size(fullySheared.output[l])[end]
                layerData[l][outer, innerAxis..., i] =
                    resample(layerOutput[l][outer, innerAxis..., i], 0.0,
                             newSize = size(layerData[l])[(end-2):(end-1)])
            end
        end 
    end
end
function estimateTheLastLayer!(layerData, layerOutput, shears, nonlin, P)
    # once we've upsampled, convolve with the dual frame-- this is not enough for recovery unless the support of the signal is strictly within the support of the averaging filter
    padBy = getPadBy(shears)
    innerAxis = axes(layerData)[(end-2):(end-1)]
    innerAxisOut = axes(layerData)[(end-2):(end-1)]
    outerAxis = axes(layerData)[1:(end-2)]
    for outer in eachindex(view(layerData, outerAxis, 1, 1,1))
        for i=1:size(layerData[end])[end]
            reconstructedData = inverseNonlin.(layerOutput[outer,
                                                           innerAxisOut..., i],
                                               nonlin)
            reconstructedData = resample(reconstructedData, 0.0, newSize =
                                         size(layerData)[(end-2):(end-1)])
            layerData[outer, innerAxis...,i] =
                shearrec(reconstructedData, shears, P, true, padBy,
                         averaging=true)
        end
    end
end

function estimateMidLayer!(m, layerData, shear, nonlin)
    innerAxis = axes(layerData[l])[(end-2):(end-1)]
    outerAxis = axes(layerData[l])[1:(end-2)]
    for outer in eachindex(view(layerData, outerAxis, 1, 1,1))
        for indexInData=0:(size(layerData)-1)
            fromNextLayerSub = inverseNonlin.(layerData[m+1][outer,
                                                             innerAxis...,
                                                             getChildren(layers,
                                                                         m)],
                                              nonlin)
            fromNextLayer = zeros(size(layerData[m][outer, innerAxis...])...,
                                  length(getChildren(layers, m)))
            for k =1:size(fromNextLayerSub,3)
                fromNextLayer[:, :, k] = resample(fromNextLayer, 0.0, newSize =
                                                  size(fromNextLayer)[1:2])
            end
            postShearlet = cat(3, fromNextLayer, layerData[m][outer,
                                                              innerAxis...,
                                                              indexInData])
            layerData[m][outer, innerAxis..., indexInData + 1] =
    shearrec2D(postShearlet, shear, P, true, padBy, averaging=true)
                resample(inverseNonlin.(shearrec2D(tmp, shear), nonlin), -1.0,
                         newSize = size(layerData)[end-3:end-2])
        end
    end
end



using Interpolations
X = example[10:24,10:24]
itp = interpolate(X, BSpline(Quadratic(Reflect(OnGrid()))))
etpf = extrapolate(itp, Line())   # gives 1 on the left edge and 7 on the right edge
etp0 = extrapolate(itp, 0)        # gives 0 everywhere outside [1,7]
plot(heatmap(etpf(-10:30, 1:(24-10+1))), heatmap(example), heatmap(X),
     layout=(1,3))

@time reconstructionSp, backShatter = pseudoInversion(fullySheared,layers,softplusType())
heatmap(flipdim(reconstructionSp,1))
heatmap(flipdim(example,1))
@time reconstructiontanh, backShatter = pseudoInversion(fullySheared,layers,tanhType())
heatmap(flipdim(reconstructiontanh,1))
heatmap(reconstructiontanh-reconstructionSp)

mostlyZeros = deepcopy(fullySheared)
for (i,x) in enumerate(mostlyZeros.output)
    mostlyZeros.output[i] = randn(size(x))
end
mostlyZeros.output[4]
mostlyZeros.output[4]=100+100im*ones(13,13,4096)
singleEntry, backShatter = pseudoInversion(mostlyZeros,layers)
heatmap(singleEntry)
minimum(singleEntry)
heatmap(abs.(fullySheared.output[end][:,:,8]))
heatmap(original-example)
heatmap(original)
heatmap(real.(reconstruction.data[4][:,:,4]))
heatmap(real.(fullySheared.data[4][:,:,4]))
layers.shears

upSample(spInverse.(reconstruction.data[m+1][:,:,k]), size(reconstruction.output[m-1])[1:2])
(numberSkipped(m,m-1,layers))
numChildren(pathType(m,[[1,1,1] for i=1:m]),layeredTransform)
# now that we have the data from the final layer, we can recursively reconstruct to get the original input, using the
heatmap(abs.(spInverse.(reconstruction.data[end][:,:,3])))
heatmap(abs.(fullySheared.data[end][:,:,3]))
convDualAveragingFilter(reconstruction.output[end][:,:,1], layers.shears[end])
fullySheared.data[4]
fullySheared.output[4]
heatmap(real.(fullySheared.data[2][:,:,1]))

# first let's see if we can reconstruct the input given the data in the first layer
fullySheared.output[1]
heatmap(real.(spInverse.(fullySheared.data[2][:,:,1])))
spInverse.(fullySheared.data[2]); fullySheared.output[1]
inputMaybe = shearrec2D(cat(3,[upSample(spInverse.(fullySheared.data[2][:,:,i]),size(reconstruction.output[1][:,:,1])) for i=1:size(fullySheared.data[2],3)]..., reconstruction.output[1][:,:,1]),layers.shears[1])
shearrec2D(sheardec2D(example, layers.shears[1]), layers.shears[1])-example
# given exactly the data in the first layer, the reconstruction is decent, though still missing some details due to subsampling
heatmap(inputMaybe)
heatmap((inputMaybe-example)./max.(1,example))


minimum(inputMaybe)
iinput = sheardec2D(example,layers.shears[1])

# checking that the upSampling did what was expected
l=3; k = 11; heatmap(real.(upsampled[l+1][:,:,k]))
heatmap(real.(fullySheared.output[l+1][:,:,k]))

heatmap(layers.shears[1].dualFrameWeights)

X = zeros(Complex{Float64},size(upsampled)...)
X += fftshift(fft(ifftshift(upsampled[:,:,1]))).*layers.shears[1].shearlets[:,:,end]
out = real(fftshift(ifft(ifftshift((1 ./layers.shears[1].dualFrameWeights).*X))))
heatmap(out)


# TEST ALGORITHM TODO think about guarantees-- probably something about both the invertibility of the nonlinearity and the spread of the signal over various frequencies
# start by inverting just the deepest layers
for i=layers.m:-1:0

end
layers.shears[1]




softplus(x::T) where T<:Real = log(1+exp(real(x))) - log(2)
softplus(x::ComplexF64) = log(1+exp(real(x))) + im*log(1+exp(imag(x)))
# inverse of the softplus function (note that it isn't accurate for input smaller than -36 because of numerical issues in the softplus function itself)
spInverse(x::T) where T<:Real = log(exp(x+log(2))-1)
spInverse(x::Float64) = x>0 ? log(exp(x)-1) : -36.737
spInverse(x::ComplexF64) = spInverse(real(x)) + im*spInverse(imag(x))
aTanh(x::ComplexF64) = atanh(real(x)) + im*atanh(imag(x))
aTanh(x::Float64) = atanh(x)

aTanh(.1+.1im)
softplus(0)
# how accurate is the inverse of tanh
t=-100:.01:100;
forward = softplus.(t);
plot(t, forward)
back = spInverse.(forward)
acc = map(x->(abs(x)==Inf ? sign(x) : x), (back-t)./abs.(t))
plot(t,acc)
# apparently it breaks down pretty badly around ±15, so that's that

# an alternative nonlinearity that is
#    smooth
#    zero at zero
#    numerically stable
#    linear at x→ ∞ with different slopes
# need to think about whether we actually need differentiability at zero
twoLines(x::T;a=1.0, b=1/16) where T<:Real = log(1 + exp(a*x)) - log(1+exp(-b*x))
twoLinesInverse(y::T;a=1.0, b=1/16) where T<:Real = 
sinh.(t)
plot(t, twoLines.(t))
