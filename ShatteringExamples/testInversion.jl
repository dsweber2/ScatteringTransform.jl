using Shearlab
using Plots
using ScatteringTransform
using MLDatasets

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

function convDualAveragingFilter(coeffs::Array{ComplexF64,2},shearletSystem::Shearlab.Shearletsystem2D)
    gpu = shearletSystem.gpu
    if gpu == 1
        X = AFArray(zeros(Complex{Float32},size(coeffs,1),size(coeffs,2)))
    else
        X = zeros(Complex{Float64},size(coeffs,1),size(coeffs,2))
    end
    X = fftshift(fft(ifftshift(coeffs))).*shearletSystem.shearlets[:,:,end]
    return real(fftshift(ifft(ifftshift((1 ./shearletSystem.dualFrameWeights).*X))))
end

# create an array of the resampled scattering transform outputs
function pseudoInversion(fullySheared::shattered{T},layers::layeredTransform, X::tanhType) where T
    # make a storage array where we will estimate the data
    backShatter = shattered{T}(layers, zeros(size(example)))
    
    # upsampled = Array{Array{T,3},1}(length(fullySheared.output))
    zerothLayerUpsample = upSample(fullySheared.output[1][:,:,1],(28,28))
    backShatter.output[1] = reshape(zerothLayerUpsample,(size(zerothLayerUpsample)...,1))
    # start by upsampling all of the outputs
    for l = layers.m+1:-1:2
      backShatter.output[l] = Array{T,3}(size(fullySheared.output[l-1][:,:,1])..., size(fullySheared.output[l],3))
      for i= 1:size(fullySheared.output[l],3)
        backShatter.output[l][:,:,i] = upSample(fullySheared.output[l][:,:,i], size(backShatter.output[l])[1:2])
      end
    end
    # once that's done, we start at the deepest layers and work our way up. In the deepest layer, we have a lossy estimate of the data, since we have no information about the higher frequencies at this layer

    # once we've upsampled, convolve with the dual frame-- this is not enough for recovery unless the support of the signal is strictly within the support of the averaging filter
    for i=1:size(backShatter.data[end],3)
        backShatter.data[end][:,:,i] = convDualAveragingFilter(backShatter.output[end][:,:,i], layers.shears[end])
    end
    for m=layers.m:-1:2
        skippedThisLayer = numberSkipped(m,m-1,layers)
        for indexInData=0:(size(backShatter.data[m],3)-1)
            size(backShatter.output[m-1])[1:2]
            tmp = cat(3, backShatter.data[m+1], backShatter.output[m+1][:,:,indexInData+1])
            backShatter.data[m][:,:,indexInData+1] = upSample(aTanh.(shearrec2D(tmp,layers.shears[m+1])), size(backShatter.output[m])[1:2])
        end
    end

    # we've worked backwards through the data; now reconstruct the input
    reconstruction = shearrec2D(cat(3,[upSample(aTanh.(backShatter.data[2][:,:,i]),size(backShatter.output[1][:,:,1])) for i=1:size(fullySheared.data[2],3)]..., backShatter.output[1][:,:,1]),layers.shears[1])
    backShatter.data[1] = reshape(reconstruction,(size(reconstruction)...,1))
    return (reconstruction,backShatter)
end
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
