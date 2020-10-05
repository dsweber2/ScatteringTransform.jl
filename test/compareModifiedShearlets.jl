

using Revise
using Shearlab
using ScatteringTransform
using Plots

using cartoonLike
using LinearAlgebra


# compare the two transforms on a random matrix
m1=2; n1=50; p1=55
X1 = randn(n1,p1)
layer1 = stParallel(m1, size(X1), typeBecomes=eltype(X1))
scatteringSheared = ScatteringTransform.sheardec2D(X1, layer1.shears[1])

shears = Shearlab.getshearletsystem2D(size(X1)..., 2)
tinyShears = Shearlab.getshearletsystem2D(50,55,2)
tinyOutput = Shearlab.SLsheardec2D(X1,tinyShears)
norm((scatteringSheared)/norm(scatteringSheared,Inf) - tinyOutput/norm(tinyOutput,Inf))
plot(heatmap(scatteringSheared[:,:,1]), heatmap(real.(tinyOutput)[:,:,1]), heatmap(real.(scatteringSheared - tinyOutput)[:,:,1]), layout=(3,1))


n = 400
points = [-.25 0;
          -.26 -.3;
           .1 -.25;
           .3  .3;
          -.3  .25]
X1,xLocs,yLocs = given_points(points, cart = cartoonLikeImage{Float64}(xSize=n, ySize=n))
heatmap(xLocs, yLocs, X1)
layer1 = stParallel(m1, size(X1), typeBecomes=eltype(X1))
scatteringSheared = ScatteringTransform.sheardec2D(X1, layer1.shears[1])

shears = Shearlab.getshearletsystem2D(size(X1)..., 2)
tinyShears = Shearlab.getshearletsystem2D(400, 400, 2)
tinyOutput = Shearlab.SLsheardec2D(X1, tinyShears)

norm((scatteringSheared)/norm(scatteringSheared,Inf) - tinyOutput/norm(tinyOutput,Inf))
# it agrees very closely on a more realistic image that isn't just gaussian noise everywhere. 
k = 7; plot(heatmap(scatteringSheared[:,:,k]), heatmap(real.(tinyOutput)[:,:,k]), heatmap(real.(scatteringSheared - tinyOutput)[:,:,k]), layout=(3,1))


subsamp = 1.5; nScales = 3
m2=2; n2=201; p2=325
X2 = randn(n2,p2)
layer2 = stParallel(m2, size(X2); subsample = subsamp, nScale = nScales,
                          typeBecomes = eltype(X2))
# A more carefully constructed test that also tests a different element type
m3=2
X3 = zeros(Float32, 1, 50, 50)
X3[1, 13:37, 13:37] = ones(Float32, 25, 25)
layer3 = stParallel(m3, size(X3)[end-1:end], typeBecomes=eltype(X3))
