using Shearlab
using Plots
using ScatteringTransform
using MLDatasets
#pyplot()

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

using Revise
using ScatteringTransform, FFTW, Plots
X3 = zeros(Float32, 1, 28, 28);
X3[1, 11:17, 11:17] = ones(Float32, 7,7)
X3 = zeros(Float32, 1, 50, 50);
X3[1, 33:41, 33:41] = ones(Float32, 9,9)
X3[1, 10:18, 10:18] = -ones(Float32, 9,9)
X3[1, 25:33, 25:33] = ones(Float32, 9,9)
heatmap(ScatteringTransform.inverseNonlin(X3[1,:,:], piecewiseType()))
savefig("reconstructSquarePiecewise.pdf")
m3= 2; layer3 = layeredTransform(m3, size(X3)[end-1:end],
                                 typeBecomes=eltype(X3), subsamples = [28/19, 19/13, 13/8])
res3 = st(X3, layer3, piecewiseType(), thin=false)
result = ScatteringTransform.pseudoInversion!(res3, layer3, piecewiseType());
plot(heatmap(9 .* X3[1,:,:]), heatmap(result.data[1][1, :, :, 1]), layout=(2,1), title="Reconstruction: Piecewise Type")
savefig("reconstructSquarePiecewise.pdf")
plot(heatmap(result.data[3][1,:,:,1]), heatmap(result.output[3][1,:,:,1]));
savefig("tmp.pdf")
res3 = st(X3, layer3, tanhType(), thin=false);
result = ScatteringTransform.pseudoInversion!(res3, layer3, tanhType())
plot(heatmap(X3[1,:,:]), heatmap(result.data[1][1, :, :, 1]),layout=(2,1),
     title="Reconstruction: Tanh Type")
savefig("reconstructSquareTanh.pdf")
res3 = st(X3, layer3, ReLUType(), thin=false);
result = ScatteringTransform.pseudoInversion!(res3, layer3, ReLUType())
plot(heatmap(X3[1,:,:]), heatmap(result.data[1][1, :, :, 1]),layout=(2,1), title="Reconstruction: ReLU Type")



X = Float32.(train_x[:,:,325])
m3= 2; layers = layeredTransform(m3, size(X)[end-1:end], typeBecomes=eltype(X),
                                 subsamples = [28/26, 26/24, 24/20])
res = st(X, layers, piecewiseType(),thin=false)
result = ScatteringTransform.pseudoInversion!(res, layers, piecewiseType())
plot(heatmap(X), heatmap(result.data[1][1, :, :, 1]),layout=(2,1), title="Reconstruction: Piecewise Type")
savefig("reconstructMNISTPiecewise.pdf")
