using Random
using Wavelets
using ContinuousWavelets
Random.seed!(135)
using Test
@testset "st using cwt tests" begin
    sharpExample = zeros(100, 2); sharpExample[26:75,:] .= 1;
    layers = stParallel(2, size(sharpExample, 1))
    resSt = st(sharpExample, layers, abs)
    @test eltype(resSt) <: eltype(sharpExample) # check the type
    otherResSt = layers(sharpExample)
    compare = roll(resSt, layers, sharpExample)
    @test compare == otherResSt
    @test compare[2][:,end,end,1] == resSt[end - 12:end,1]
    multiDExample = 1000 * randn(50, 10, 43)
    layers = stParallel(2, size(multiDExample, 1))
    resSt = st(multiDExample, layers, abs);
    @test eltype(resSt) <: eltype(multiDExample)
    @test 0 < minimum(abs.(resSt[:, 1, 1]))  # input is always non-zero so nothing should be *Exactly* zero, except by chance (which is why we set the seed above)
    @test 0 < minimum(abs.(resSt[:, end, end]))
    resSt = st(multiDExample, layers, abs, outputSubsample=(3, -1))
    @test eltype(resSt) <: eltype(multiDExample)
    @test 0 < minimum(abs.(resSt[:,1,1]))  # input is always non-zero so nothing should be *Exactly* zero, except by chance (which is why we set the seed above)
    @test 0 < minimum(abs.(resSt[:, end,end]))
end


f = randn(102)
lay = stParallel(1, size(f)[1])
ScatteredFull(lay, f)
# some basic tests for the Scattered type
# m=2, 3extra, 2transformed
fixDims = [3,5,2]
m = 2; k = 2
n = [100 200; 50 100; 25 50]; q = [5, 5, 5]
ex1 = ScatteredFull(m, k, fixDims, n, q, Float64)
@test ex1.m == m
@test ex1.k == k
@test size(ex1.data, 1) == m + 1
@test size(ex1.output, 1) == m + 1
@testset "correct size for output" begin
    for i = 1:size(ex1.output, 1)
        @test size(ex1.output[i])[4:end] == tuple(fixDims...)
        @test size(ex1.data[i])[4:end] == tuple(fixDims...)
        @test size(ex1.output[i])[1:2] == tuple(n[i,:]...)
        @test size(ex1.data[i])[1:2] == tuple(n[i,:]...)
    end
end
