########################### inversion tests ##############################


using ScatteringTransform, FFTW 
X3 = zeros(Float32, 1, 50, 50);
X3[1, 13:37, 13:37] = ones(Float32, 25, 25);
m3= 2; layer3 = layeredTransform(m3, size(X3)[end-1:end], typeBecomes=eltype(X3));
n, q, dataSizes, outputSizes, resultingSize = ScatteringTransform.calculateSizes(layer3,
                                                                                 (-1,-1),
                                                                                 size(X3))
fftPlans = createFFTPlans(layer3, dataSizes, iscomplex = false)
fftPlan = fetch(fftPlans[1]);
ScatteringTransform.getPadBy(layer3.shears[1])
shears = layer3.shears[1];
size(X3)
coeffs = ScatteringTransform.sheardec2D(view(X3,:,:,:), layer3.shears[1],fftPlan,true, (47,49));
reconstruct = ScatteringTransform.shearrec2D(view(coeffs, :,:,:,:), layer3.shears[1], fftPlan, true, (47,49), (50,50))
using Plots
res3 = st(X3, layer3, piecewiseType(), thin=false);
result = ScatteringTransform.pseudoInversion!(res3, layer3, piecewiseType())
plot(heatmap(X3[1,:,:]), heatmap(result.data[1][1, :, :, 1]),layout=(2,1), title="Reconstruction: Piecewise Type")
res3 = st(X3, layer3, absType(), thin=false);
result = ScatteringTransform.pseudoInversion!(res3, layer3, absType())
plot(heatmap(X3[1,:,:]), heatmap(result.data[1][1, :, :, 1]),layout=(2,1), title="Reconstruction: Abs Type")
res3 = st(X3, layer3, ReLUType(), thin=false);
result = ScatteringTransform.pseudoInversion!(res3, layer3, ReLUType())
plot(heatmap(X3[1,:,:]), heatmap(result.data[1][1, :, :, 1]),layout=(2,1), title="Reconstruction: ReLU Type")

testScattered = scattered(layer3, X3)

@testset "inversion tests" begin
    @test typeof(reconstruct) <: typeof(X3)
    @test norm(X3-reconstruct,2)<.001 # the error in the reconstruction should be less than .1%
end
