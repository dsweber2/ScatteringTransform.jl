@testset "st using cwt tests" begin
    sharpExample = zeros(100); sharpExample[26:75] = ones(75 - 25)
    layers = layeredTransform(2, size(sharpExample, 1))
    resSt = st(sharpExample, layers, absType())
    @test eltype(resSt) <: eltype(sharpExample) # check the type
    multiDExample = randn(5, 10, 43)
    resSt = st(multiDExample, layers, absType())
    @test eltype(resSt) <: eltype(multiDExample)
    @test 0 < minimum(abs.(resSt))  # input is always non-zero so nothing should be *Exactly* zero
    resSt = st(multiDExample, layers, absType(), outputSubsample=(3,-1))
    @test eltype(resSt) <: eltype(multiDExample)
    @test 0 < minimum(abs.(resSt))  # input is always non-zero so nothing should be *Exactly* zero
end


f = randn(102)
lay = layeredTransform(1, size(f)[1])
scattered(lay, f)
# some basic tests for the scattered type
# m=2, 3extra, 2transformed
fixDims = [3,5,2]
m = 2; k=2
n = [100 200; 50 100; 25 50]; q = [5, 5, 5]
ex1 = scattered(m, k, fixDims, n, q,Float64)
@test ex1.m==m
@test ex1.k==k
@test size(ex1.data,1)==m+1
@test size(ex1.output,1)==m+1
@testset "correct size for output" begin
  for i=1:size(ex1.output,1)
      @test size(ex1.output[i])[1:size(fixDims,1)] == tuple(fixDims...)
      @test size(ex1.data[i])[1:size(fixDims,1)] == tuple(fixDims...)
      @test size(ex1.output[i])[size(fixDims,1)+1:end-1] == tuple(n[i,:]...)
      @test size(ex1.data[i])[size(fixDims,1)+1:end-1] == tuple(n[i,:]...)
  end
end

################################ cwt tests ################################
# 1D input tests
@testset "cwt tests" begin
    inputs = [randn(1, 128), randn(100,128), randn(10, 21, 128), randn(2, 3, 5, 128)]
    averagingTypes = [:Mother, :Dirac]
    scalingfactors = reverse([1/2, 1, 2, 8, 16, 32])
    waves = [WT.morl WT.dog2 WT.paul4]
    precomputed =[true, false]
    for inn in inputs
        for wave in waves
            for sb in scalingfactors
                for ave in averagingTypes
                    for comp in precomputed
                        waveConst = CFW(wave, sb, ave)
                        println("$(wave), $(sb), $(ave), $(size(inn))")
                        if comp
                            daughters = computeWavelets(size(inn,1), waveConst)
                            println("WAT, $(typeof(inn)), $(typeof(waveConst)), $(typeof(daughters))")
                            output = cwt(inn, waveConst, daughters)
                        else
                            output = cwt(inn, waveConst)
                        end
                        @test numScales(waveConst,size(inn)[end])+1 ==
                            size(output)[end]
                        @test length(size(output))== 1+length(size(inn))
                        # testing just averaging
                        if comp
                            daughters = computeWavelets(inn, waveConst,
                                                        nScales=0)
                            output = cwt(inn, waveConst, daughters, nScales=0)
                        else
                            output = cwt(inn, waveConst, nScales=0)
                        end
                        @test size(output)[end]==1
                        @test length(size(output))== 1+length(size(inn))
                        if comp
                            daughters = computeWavelets(inn,waveConst,
                                                        nScales=20)
                            output = cwt(inn,waveConst, daughters, nScales=20)
                        else
                            output = cwt(inn,waveConst,nScales=20)
                        end
                        @test 21 == size(output)[end]
                        @test length(size(output))== 1+length(size(inn))
                    end
                end
            end
        end
    end
end


