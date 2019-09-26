using Revise
using ScatteringTransform, FFTW, Plots, LinearAlgebra
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
norm(result.data[1][1, :, :, 1]- X3[1,:, :], 2)./norm(X3,2)
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



train_x, train_y = MNIST.traindata();
XM = Float32.(train_x[:,:,325])
m3= 2; layers = layeredTransform(m3, size(X)[end-1:end], typeBecomes=eltype(X),
                                 subsamples = [28/26, 26/24, 24/20])
resM = st(XM, layers, piecewiseType(),thin=false)
resultM = ScatteringTransform.pseudoInversion!(resM, layers, piecewiseType())
pyplot()
plot(heatmap(reverse(XM',dims=1)), heatmap(reverse(result.data[1][1, :, :, 1]', dims=1)),layout=(2,1), title="Reconstruction: Piecewise Type");
savefig("reconstructMNISTPiecewise.pdf")


train_x, train_y = FashionMNIST.traindata();
X = Float32.(train_x[:,:,700]); heatmap(reverse(X',dims=1),color=:Greys, aspect_ratio=1); savefig("tmp.pdf")
m3= 2; layers = layeredTransform(m3, size(X)[end-1:end], typeBecomes=eltype(X),
                                 subsamples = [28/26, 26/24, 24/20])
res = st(X, layers, piecewiseType(),thin=false)
result = ScatteringTransform.pseudoInversion!(res, layers, piecewiseType())
pyplot()
plot(heatmap(reverse(X',dims=1), aspect_ratio=1, colorbar=false, color=:Greys), heatmap(reverse(result.data[1][1, :, :, 1]', dims=1), colorbar=false, color=:Greys, aspect_ratio=1),layout=(2,1));
savefig("reconstructFashionMNISTPiecewise.pdf")

# plot them all on the same plot
plot(heatmap(reverse(XM',dims=1), aspect_ratio=1, colorbar=false,
             color=:Greys,title="Original 7"), heatmap(reverse(X',dims=1),
                                                       aspect_ratio=1,
                                                       colorbar=false,
                                                       color=:Greys,
                                                       title="Original Sneaker"),
     heatmap(reverse(resultM.data[1][1, :, :, 1]', dims=1), 
             colorbar=false, color=:Greys, aspect_ratio=1, title="Recovered 7"),
     heatmap(reverse(result.data[1][1, :, :, 1]', dims=1),
             colorbar=false, color=:Greys, aspect_ratio=1, title="Recovered Sneaker"), layout=(2,2));
savefig("reconstructPiecewise.pdf")
norm(XM-resultM.data[1][1, :, :, 1])/norm(XM)
norm(X-result.data[1][1, :, :, 1])/norm(X)

using HDF5
fn = "/VastExpanse/workingDataDir/shattering/MNIST/finalCoeffs.h5"
h5file = h5open(fn, "r")
coeffs = Float32.(h5file["piecewise"][:,:]')
close(h5file)
layers = layeredTransform(2, (28, 28), subsamples=[28/19, 19/13, 13/8], shearLevel=Int.(ceil.((1:4)/4)), typeBecomes = Float32);
tmpThing = st(X, layers, piecewiseType());

for i=1:size(coeffs,1)
    println("i=$(i)")
    pos = ScatteringTransform.wrap(layers, reshape(coeffs[i, :],
                                                   (1,size(coeffs,2))),
                                   zeros(Float32, 1, 28,28))
    neg = ScatteringTransform.wrap(layers, reshape(-coeffs[i, :],
                                                   (1,size(coeffs,2))),
                                   zeros(Float32, 1, 28,28))
    resPos  = ScatteringTransform.pseudoInversion!(pos, layers,
                                                   piecewiseType())
    resNeg  = ScatteringTransform.pseudoInversion!(neg, layers,
                                                   piecewiseType())
    pyplot(); heatmap(resPos.data[1][1,:,:,1] + resNeg.data[1][1,:,:,1], aspect_ratio=1, color=:Greys)
    savefig("coeffPlots/spaceDiff$(i).pdf")
    pyplot(); heatmap(resPos.data[1][1,:,:,1], aspect_ratio=1, color=:Greys)
    savefig("coeffPlots/coeff$(i).pdf")
    pyplot(); heatmap(resNeg.data[1][1,:,:,1], aspect_ratio=1, color=:Greys)
    savefig("coeffPlots/NegCoeff$(i).pdf")
end


1325
100 # a simple 1
200 # a two
400 # a zero
X = Float32.(train_x[:,:,400]); heatmap(X); savefig("tmp.pdf")
# a point halfway between a 1 and a zero in scattering space
X0 = Float32.(train_x[:,:,400]);
X1 = Float32.(train_x[:,:,100]);
s1 = st(X1, layers, piecewiseType()); s0 = st(X0, layers, piecewiseType());
function pathBetween(s0, s1, t, layers)
    wrap01 = ScatteringTransform.wrap(layers, s1*t + s0*(1-t), zeros(Float32, 1, 28, 28))
    resAve01 = ScatteringTransform.pseudoInversion!(wrap01, layers,
                                                    piecewiseType())
    return resAve01
end
function plotPath(s0,s1,T,layers, scattered=true)
    ht = Array{Any}(undef,length(T))
    for (ti, t) in enumerate(T)
        if scattered
            ht[ti] = heatmap(reverse(pathBetween(s0,s1,t, layers).data[1][1, :,
                                                                          :,
                                                                          1]',
                                     dims=1), title="t=$(t)",color=:Greys)
        else
            ht[ti] = heatmap(reverse((s0*(1-t) + s1*t)', dims=1),
                             title="t=$(t)",color=:Greys)
        end
    end
    plot(ht...)
end

X2 = Float32.(train_x[:, :, 200])
X7 = Float32.(train_x[:, :, 1325])
plotPath(st(X2, layers, piecewiseType()), st(X7, layers, piecewiseType()),
         0:.125:1, layers)
savefig("coeffPlots/path25.pdf")
plotPath(X2,X7, 0:.125:1, layers, false)
savefig("coeffPlots/path25NORMAL.pdf")
X0 = Float32.(train_x[:, :, 2023])
X1 = Float32.(train_x[:, :, 4235])
plotPath(st(X0, layers, piecewiseType()), st(X1, layers, piecewiseType()),
         0:.125:1, layers)
savefig("coeffPlots/path23.pdf")
plotPath(X0,X1, 0:.125:1, layers, false)
savefig("coeffPlots/path23NORMAL.pdf")


plot(heatmap(reverse(X0',dims = 2)), heatmap(reverse(pathBetween(s0,s1,.9).data[1][1, :, :,
                                                                    1]', dims =
                                                   2)),heatmap(reverse(X1',
                                                                       dims = 2)))
savefig("coeffPlots/ave01.pdf")


X0 = Float32.(train_x[:,:,400]);
X1 = Float32.(train_x[:,:,100]);
s1 = st(X1, layers, piecewiseType()); s0 = st(X0, layers, piecewiseType());









hbase1 = heatmap(reverse(X1', dims=1)); savefig("tmp.pdf")
hbase0 = heatmap(reverse(X0', dims=1)); savefig("tmp.pdf")
# what about a particular X1 separates it from an X0?
reweighted0 = ScatteringTransform.wrap(layers, reshape(coeffs[1,:] .* s0',
                                                       (1,size(coeffs,2))),
                                       zeros(Float32, 1, 28,28))

resAve0 = ScatteringTransform.pseudoInversion!(reweighted0, layers,
                                               piecewiseType())
h0 = heatmap(reverse(resAve0.data[1][1, :, :, 1]', dims=1),color=:Greys)
savefig("tmp.pdf")

reweighted1 = ScatteringTransform.wrap(layers, reshape(coeffs[1,:] .* s1',
                                                       (1,size(coeffs,2))),
                                       zeros(Float32, 1, 28,28))

resAve1 = ScatteringTransform.pseudoInversion!(reweighted1, layers,
                                               piecewiseType())
h1 = heatmap(reverse(resAve1.data[1][1, :, :, 1]', dims=1),color=:Greys)


pos = ScatteringTransform.wrap(layers, reshape(coeffs[i, :],
                                               (1,size(coeffs,2))),
                               zeros(Float32, 1, 28,28))
resPos  = ScatteringTransform.pseudoInversion!(pos, layers,
                                               piecewiseType())




savefig("tmp.pdf")


# looking at a specific coordinate
function plotCoord(m,j,onesAtx,onesAty=onesAtx)
    if typeof(onesAtx)<:Int
        onesAtx = onesAtx:onesAtx
    end
    if typeof(onesAty)<:Int
        onesAty = onesAty:onesAty
    end
    singleCoord = ScatteringTransform.wrap(layers, zeros(Float32, 1, 10340),
                                           zeros(Float32, 1, 28,28))
    singleCoord.output[m+1][1, onesAtx, onesAty, j] = ones(onesAtx,onesAty)
    image = ScatteringTransform.pseudoInversion!(singleCoord, layers,
                                                 piecewiseType());
    return image.data[1][1,:,:,1]
end
heatmap(plotCoord(2, 37, 4, 5))



using FFTW
equivShear = zeros(Float32, 80, 82,17); equivShear[39:42, 40:43, 1] =
    ones(4,4)*.5
P = plan_rfft(equivShear);
equivShear[40:41, 41:42, 1] = ones(2,2)
size(P)

