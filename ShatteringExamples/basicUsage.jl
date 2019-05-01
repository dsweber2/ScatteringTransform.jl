# I suggest running this line by line to see the figures
using Plots, LinearAlgebra
using MLDatasets
using ScatteringTransform
using Shearlab, FFTW
# MNIST examples
train_x, train_y = MNIST.traindata()
Y0 = Float32.(reshape(train_x[:, :, 2],(28,28))); Y0 = Y0 ./norm(Y0);
Y1 = Float32.(reshape(train_x[:, :, 4],(28,28))); Y1 = Y1 ./norm(Y1);
Y2 = Float32.(reshape(train_x[:, :, 6],(28,28))); Y2 = Y2 ./norm(Y2);
heatmap(Y0)
heatmap(randn(100,100))
heatmap(Float64.(reverse(Y0', dims=1))) # a zero
heatmap(reverse(Y1', dims=1)) # a one
heatmap(reverse(Y2', dims=1)) # a two
# number of layers
m = 2
# set the number of scales
sc = 2
# set the amount of shearing; it scales as 2^shearLevels
shearLevels = 3
# set the subsampling rate
subsamp = 28/24
layers = layeredTransform(m, size(Y0); nScale=sc, shearLevel=[shearLevels for i=1:m+1], subsample=subsamp)
@time Yshattered0 = st(Y0, layers, absType())
@time Yshattered1 = st(Y1, layers, absType())
@time Yshattered2 = st(Y2, layers, absType())
# focusing on the zero for a bit
plotShattered(Yshattered0, layers) #shows the total energy in each path in the second layer
savefig("shattered.pdf")
# looks like cone 1, scale 2, shear 0 [1,2,0] in layer one followed by cone 1, scale 1 and shear 1 [1,1,1] in layer 2 has the highest energy. Let's take a closer look at that one
path = pathType(2, [[1,2,0], [1,1,1]])
plotCoordinate(path, 2, layers, Yshattered0)
# looking at the Fourier domain shearlets corresponding to this shattered path:
plotCoordinateShearlet(path, layers)
# and in the time domain
plotCoordinate(path, layers, fun=x->abs.(fftshift(fft(x))))
# The second layer is picking up on the slant to the right
plotCoordinate(path, 2, layers, Yshattered0)

Yshattered0.data[2]
heatmap(Yshattered0.data[3][:,:,7])
heatmap(Yshattered0.output[1][:,:,1])
heatmap(Yshattered0.output[2][:,:,3])
# Shattering transforms can be treated as vectors, of a sort
Yshattered1-Yshattered0
norm(Yshattered0)
norm(Yshattered1)
norm(Yshattered2)
norm(Yshattered0-Yshattered1)
norm(Yshattered1-Yshattered2)
norm(Yshattered0-Yshattered2)
# so in the shattered domain this zero and this two are closer together than this 1 and this 2, though not by much. This is also true in the original domain:
norm(Y0-Y1)
norm(Y1-Y2)
norm(Y0-Y2)
heatmap(Y0)
plotShattered(Yshattered0,layers,scale=:none) #the zero is highly concentrated at [1,2,0],[1,1,1]
heatmap(Y1)
plotShattered(Yshattered1,layers,scale=:none) # the one is smeared around the same region
plotShattered(Yshattered2,layers,scale=:none) # the two has more horizontal components, but also a large component from [1,2,0],[1,1,1]
heatmap(reverse(Y2',dims=1))

# what kind of input then does [1,2,0],[1,1,1] correspond to?


norm(Yshattered1)
norm(Yshattered0)
norm(Yshattered2)
Y = Y1
# lets look at a different nonlinearity; the coefficients output here are complex
@time YReLU = shatter(Y,layers,ReLUType())
plotShattered(YReLU,layers,scale=:none)
plotShattered(Yshattered1,layers,scale=:none) #the zero is highly concentrated at [1,2,0],[1,1,1]
# there is
plts = Array{Plots.Plot}(3)
plts[1] = plt
plts[2] = plt2
plts[3] = plt
plot(plts...,layout=(3,1))
# and let's see what the shears used were
plotCoordinate
path = pathType(2, [13, 12], layers) #constructs a path object using linear indices
path = pathType()
path =pathType(2, [[1,1,1],[1,2,1]])
# alteratively, if you want the earlier levels to have more scales and the later levels to subsample more
sc = [4,2,1]
subsamp = [1.0,1.5,2]
layers = layeredTransform(m,Y;nScale=sc,subsample=28/24)


shears = Shearlab.getshearletsystem2D(size(Y)...,2)
tinyShears = Shearlab.getshearletsystem2D(28,28,4)
tinyOutput = Shearlab.SLsheardec2D(MNISTI,tinyShears)
n=0
orig = Shearlab.SLshearrec2D(tinyOutput,tinyShears)
heatmap(orig-MNISTI)
plt0 = heatmap(abs.(MNISTI))
n+=1; plt1 = heatmap(abs.(tinyOutput[:,:,n]),title="$n,$(tinyShears.shearletIdxs[n,:])"); plt2 = heatmap(abs.(tinyShears.shearlets[:,:,n])); plot(plt0,plt2,plt1,layout=(1,3))

tinyOutputtmp = deepcopy(tinyOutput)
tinyOutputtmp[abs.(tinyOutputtmp).<=1e-10] = 0
tinyOutputSp = [sparse(tinyOutputtmp[:,:,i]) for i=1:size(tinyOutput,3)]
28^2
nnz(tinyOutputSp[11])
sparse(tinyOutput[:,:,1])

output = Shearlab.SLsheardec2D(X, shears)
outputtmp = deepcopy(output)
outputtmp[abs.(outputtmp).<=] = 0
outputSp = [sparse(outputtmp[:,:,i]) for i=1:size(output,3)]

[nnz(outputSp[i]) for i=1:length(outputSp)]./641/481
heatmap(abs.(output[:,:,17]))
# default subsampling rate is 2.0
layers = layeredTransform(m, X, nScale)

# default non-linearity is absolute value, and is set at transform time
output = shatter(X,layers)
# plot the total magnitude of each path, indexed frist vertically, then horizontally
plotShattered(output, layers)
# thus the path ([2,1,0], [2,1,0]) has a large magnitude. To plot this particular coordinate, it is the 12th entry in each list, so
plotCoordinate([[2,1,0], [2,1,0]],2, layers, output)
