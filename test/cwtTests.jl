using Random
using Wavelets
Random.seed!(135)
@testset "st using cwt tests" begin
    sharpExample = zeros(100,2); sharpExample[26:75,:] .= 1;
    layers = stParallel(2, size(sharpExample, 1))
    resSt = st(sharpExample, layers, absType())
    @test eltype(resSt) <: eltype(sharpExample) # check the type
    multiDExample = 1000*randn(50, 10, 43)
    layers = stParallel(2, size(multiDExample, 1))
    resSt = st(multiDExample, layers, absType());
    @test eltype(resSt) <: eltype(multiDExample)
    @test 0 < minimum(abs.(resSt[:, 1, 1]))  # input is always non-zero so nothing should be *Exactly* zero, except by chance (which is why we set the seed above)
    @test 0 < minimum(abs.(resSt[:, end, end]))
    resSt = st(multiDExample, layers, absType(), outputSubsample=(3,-1))
    @test eltype(resSt) <: eltype(multiDExample)
    @test 0 < minimum(abs.(resSt[:,1,1]))  # input is always non-zero so nothing should be *Exactly* zero, except by chance (which is why we set the seed above)
    @test 0 < minimum(abs.(resSt[:, end,end]))
end


f = randn(102)
lay = stParallel(1, size(f)[1])
Scattered(lay, f)
# some basic tests for the Scattered type
# m=2, 3extra, 2transformed
fixDims = [3,5,2]
m = 2; k=2
n = [100 200; 50 100; 25 50]; q = [5, 5, 5]
ex1 = Scattered(m, k, fixDims, n, q,Float64)
@test ex1.m==m
@test ex1.k==k
@test size(ex1.data,1)==m+1
@test size(ex1.output,1)==m+1
@testset "correct size for output" begin
  for i=1:size(ex1.output,1)
      @test size(ex1.output[i])[4:end] == tuple(fixDims...)
      @test size(ex1.data[i])[4:end] == tuple(fixDims...)
      @test size(ex1.output[i])[1:2] == tuple(n[i,:]...)
      @test size(ex1.data[i])[1:2] == tuple(n[i,:]...)
  end
end

################################ cwt tests ################################
# 1D input tests
@testset "cwt tests" begin
    inputs = [randn(128, 1), randn(128,100), randn(128, 21, 10), randn(128, 2, 3, 5)]
    averagingTypes = [(WT.Father(), 2), (WT.Dirac(), 2), (WT.NoAve(), 0)]
    scalingfactors = reverse([1/2, 1, 2, 8, 16, 32])
    waves = [WT.morl WT.dog2 WT.paul4]
    precomputed =[true, false]
    for inn in inputs
        for wave in waves
            for sb in scalingfactors
                for (ave, aveLen) in averagingTypes
                    for comp in precomputed
                        waveConst = CFW(wave, sb, WT.DEFAULT_BOUNDARY, ave, aveLen)
                        println("$(wave), $(sb), $(ave), $(size(inn))")
                        if comp
                            daughters,ω = Wavelets.computeWavelets(size(inn,1), waveConst)
                            output = cwt(inn, waveConst, daughters)
                        else
                            output = cwt(inn, waveConst)
                        end
                        @test numScales(waveConst,size(inn,1)) ==
                            size(output,2)
                        @test length(size(output))== 1+length(size(inn))
                        # testing just averaging
                        if comp && aveLen > 0
                            daughters, ω = ScatteringTransform.computeWavelets(inn, waveConst,
                                                                               nScales=1)
                            output = cwt(inn, waveConst, daughters)
                            #println("size(output)= $(size(output)), size(daughters) = $(size(daughters))")
                            @test size(output,2) == 1
                        end
                    end
                end
            end
        end
    end
end


