using Shearlab
using Plots
using ScatteringTransform
using MLDatasets
pyplot()

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

