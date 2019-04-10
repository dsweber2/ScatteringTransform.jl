# tests for the various forms of layeredTransform for the ScatteringTransform
using Debugger, JuliaInterpreter
using Revise
using Interpolations, Wavelets
using FFTW
#using ScatteringTransform
using Pkg; Pkg.add("JuliaInterpreter")
exit()
using Test
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
                        waveConst = CFWA(wave, scalingFactor=sb, averagingType
                                         = ave)
                        println("$(wave), $(sb), $(ave), $(size(inn))")
                        if comp
                            daughters = computeWavelets(inn,waveConst)
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

######################################################################################################
###################################### Shattering Tests ##############################################
######################################################################################################
function testLayerConstruction(layer::layeredTransform, X::Array{Float64,2},
                               m::Int64, xSubsampled::Array{Int64},
                               ySubsampled::Array{Int64}, subsampling::Vector{Float64})
    @test layer.m==m
    @test layer.subsampling == subsampling
    @test sizes(bsplineType(), layer.subsampling,size(X,1)) == xSubsampled
    @test sizes(bsplineType(), layer.subsampling,size(X,2)) == ySubsampled
    @test sizes(bsplineType(), layer.subsampling,size(X,1))[1:end-1] ==
        [x.size[1] for x in layer.shears]
    @test sizes(bsplineType(), layer.subsampling,size(X,2))[1:end-1] ==
        [x.size[2] for x in layer.shears]
end

m1=2; n1=50; p1=55
X1 = randn(n1,p1)
layer1 = layeredTransform(m1, size(X1), typeBecomes=eltype(X1))
subsamp = 1.5; nScales = 3
m2=2; n2=201; p2=325
X2 = randn(n2,p2)
layer2 = layeredTransform(m2, size(X2); subsample = subsamp, nScale = nScales,
                          typeBecomes = eltype(X2))
# A more carefully constructed test that also tests a different element type
X3 = zeros(Float32, 100,100)
X3[26:75, 26:75] = ones(Float32, 50,50)
layer3 = layeredTransform(m1,size(X3), typeBecomes=eltype(X3))


@testset "layer construction" begin
    xSubsampled = [50, 25, 13, 7]; ySubsampled = [55, 28, 14, 7]
    testLayerConstruction(layer1, X1, m1, xSubsampled, ySubsampled, [2.0 for
                                                                     i=1:m1+1])
    xSubsampled = [201, 134, 90, 60]; ySubsampled= [325, 217, 145, 97]
    testLayerConstruction(layer2, X2, m2, xSubsampled, ySubsampled, [subsamp
                                                                     for i =
                                                                     1:m2+1])
end


################# Subsampling methods #################
function testSubsampling(X::Array{Float64},layer::layeredTransform,m::Int64)
    tmpX = Array{ComplexF64}(X)
    tmpSizesx = sizes(bsplineType(), layer.subsampling,size(X,1))
    tmpSizesy = sizes(bsplineType(), layer.subsampling,size(X,2))
    for i=1:m1+1
      @test size(tmpX)==(tmpSizesx[i],tmpSizesy[i])
      tmpX = resample(tmpX, layer.subsampling[i])
    end
end
@testset "sizes in the layered transform match those produced by subsampling" begin
    testSubsampling(X1,layer1,m1)
    testSubsampling(X2,layer2,m2)
end

#TODO: test different percentages
function testShattered(X::Array{T},layer::layeredTransform,m::Int64) where {T
                                                                            <:
                                                                            Real}
    thing = scattered(layer, X)
    @test typeof(thing.data[1]) == Array{T, 3}
    @test typeof(thing.output[1]) == Array{T, 3}
    @test thing.m ==layer.m
    # test the subsampling of the array is correct
    @test [size(thing.data[i],1) for i=1:m+1] == sizes(bsplineType(),
                                                       layer.subsampling,
                                                       size(X,1))[1:3]
    @test [size(thing.data[i],2) for i=1:m+1] == sizes(bsplineType(),
                                                       layer.subsampling,
                                                       size(X,2))[1:3]
    @test [size(thing.output[i],1) for i=1:m+1] ==
        getResizingRates(layer.shears, layer.m)[1, :]
    @test [size(thing.output[i],2) for i=1:m+1] ==
        getResizingRates(layer.shears, layer.m)[2, :]
    # check the number of scales
    q = [1; [(layer.shears[i].nShearlets-1) for i=1:m+1]]
    @test [size(thing.data[i],3) for i=1:m+1] == [prod(q[1:i]) for i=1:m+1]
    @test [size(thing.output[i],3) for i=1:m+1] == [prod(q[1:i]) for i=1:m+1]
end

@testset "shattered construction" begin
    # type input works properly
    testShattered(X1, layer1, m1)
    testShattered(X2, layer2, m2)
end

# test the nonlinearity definitions
t = [x+im*y for x=-10:1/5:10, y=-10:1/5:10]
@testset "nonlinearity definitions" begin
    @test ScatteringTransform.abs.(t)==abs.(t)
    @test ScatteringTransform.ReLU.(t) == max.(0,real(t))+im*max.(0,imag(t))
    @test ScatteringTransform.Tanh.(t) == tanh.(real(t))+im*tanh.(imag(t))
    @test ScatteringTransform.softplus.(t) == ((log.(1 .+ exp.(real.(t)))
                                                .-log(2)) + im .*(log.(1 .+
                                                                       exp.(imag.(t)))
                                                                  .-log(2)))
end
function testShattering(X::Array{Float64}, nonlinear::S, layer::layeredTransform, m::Int64) where S <: nonlinearity
    @time outputFull = shatter(X,layer,nonlinear,thin=false)
    @test typeof(outputFull.data[1])==Array{Float64,3}
    @test typeof(outputFull.output[1])==Array{Float64,3}
    @test outputFull.m ==layer.m
    # test the subsampling of the array is correct
    @test [size(outputFull.data[i],1) for i=1:m+1] == sizes(bsplineType(), layer.subsampling, size(X,1))[1:3]
    @test [size(outputFull.data[i],2) for i=1:m+1] == sizes(bsplineType(), layer.subsampling, size(X,2))[1:3]
    @test [size(outputFull.output[i],1) for i=1:m+1] == layer.reSizingRates[1, :]
    @test [size(outputFull.output[i],2) for i=1:m+1] == layer.reSizingRates[2, :]
    # check the number of scales
    q = [1; [(layer.shears[i].nShearlets-1) for i=1:m+1]]
    @test [size(outputFull.data[i],3) for i=1:m+1] == [prod(q[1:i]) for i=1:m+1]
    @test [size(outputFull.output[i],3) for i=1:m+1] == [prod(q[1:i]) for i=1:m+1]

    # check that the data isn't NaN
    @test minimum([isfinite(maximum(abs.(outputFull.data[i]))) for i=1:m+1])
    @test minimum([isfinite(maximum(abs.(outputFull.output[i]))) for i=1:m+1])

    # see how the thin version is doing
    @time outputThin = shatter(X,layer,nonlinear,thin=true)
    @test isfinite(maximum(abs.(outputThin)))

    # compare the two
    @test maximum(abs.(flatten(outputFull)-outputThin))==0.0
    return outputFull
end
@testset "actually shattering" begin
    testShattering(X1, absType(), layer1, 2)
    testShattering(X2, absType(), layer2, 2)
    testShattering(X1, ReLUType(), layer1, 2)
    testShattering(X2, ReLUType(), layer2, 2)
    testShattering(X1, tanhType(), layer1, 2)
    testShattering(X2, tanhType(), layer2, 2)
    testShattering(X1, softplusType(), layer1, 2)
    testShattering(X2, softplusType(), layer2, 2)
end




















# TODO: integrate these test in some way
# plot(heatmap(abs.((cwt(reshape(f,(1,size(f)...)), layers.shears[1],nScales=50))[1,:,:])'),heatmap(abs.(cwt(f))[5:end,:]))
# cwt(f)
# c = CFW(WT.Morlet())
# ω = [0:ceil(Int, 200/2); -floor(Int,200/2)+1:-1]*2π
# daughters = zeros(200,54)
# for a1 in 0:53
#   daughters[:,a1+1] = WT.Daughter(c, 2.0^(a1/c.scalingFactor), ω)
# end
# heatmap((daughters./[norm(daughters[:,i]) for i=1:54]')')
# norm(daughters[:,53])
# localDaughters = computeWavelets(f, layers.shears[1],nScales=50)
# layers.shears[1]
# daughters[:,5:end]'
# abs.(localDaughters[:,2:end])'
# heatmap(daughters[:,5:end]')
# heatmap(abs.(localDaughters[:,2:end])')
# (daughters[:,5:end]-localDaughters)
# heatmap(abs.(localDaughters'))
# heatmap(abs.(computeWavelets(f, layers.shears[1],nScales=49))')










function testConstruction1D(lay::layeredTransform, m::Int, shearType, averagingLength, averagingType, nShears::Vector{Int}, subsampling::Array{Float64,1}, n::Int)
  @test lay.m == m
  @test lay.subsampling==subsampling
  sizes = ScatteringTransform.sizes(bsplineType(), bspline,lay.subsampling,n)
  for i=1:length(lay.shears)
    @test typeof(lay.shears[i]) == shearType[i]
    @test lay.shears[i].averagingLength == averagingLength[i]
    @test lay.shears[i].averagingType == averagingType[i]
    @test numScales(lay.shears[i], sizes[i]) == nShears[i]
  end
end
n = 10214
f = randn(n)
lay = layeredTransform(1,f)
m=3
lay = layeredTransform(m, f, subsampling=[8,4,2,1], nScales=[16,8,8,8], CWTType=WT.dog2, averagingLength=[16,4,4,2],averagingType=[:Mother for i=1:(m+1)],boundary=[WT.DEFAULT_BOUNDARY for i=1:(m+1)])
@testset "construction tests" begin
  testConstruction1D(lay, 3, [ScatteringTransform.CFWA{Wavelets.WT.PerBoundary} for i=1:4], [16, 4, 4, 2], [:Mother for i=1:4], [165,62,46,40], [8,4,2,1]*1.0, n)
end
function testTransform(f::Array{T}, lay::layeredTransform) where T<:Number
  n=size(f)[end]
  @time output = st(f,lay)
  sizes = ScatteringTransform.sizes(bsplineType(), bspline,lay.subsampling,n)
  actualSizes = fill(0, size(sizes))
  actualSizes[1] = size(output.data[1])[end-1]
  for i=1:length(actualSizes)-2
    @test size(output.data[i+1])[end-1] == size(output.output[i])[end-1]
    actualSizes[i+1] = size(output.data[i+1])[end-1]
  end
  actualSizes[end] = size(output.output[end])[end-1]
  @test actualSizes== sizes
  # check the number of paths grows correctly
  effectiveSize = length(size(f))==1 ? (1,size(f)[1]) : size(f)
  numTransformed = ScatteringTransform.getQ(lay, effectiveSize)
  @test numTransformed[2:end]==([size(out)[end] for out in output.output])[2:end]
  @test eltype(output.data) == Array{Complex{eltype(f)},1+length(effectiveSize)}
  @test eltype(output.output) == Array{eltype(f),1+length(effectiveSize)}
end
@testset "transform tests" begin
  f=1.0 .*[1:50; 50:-1:1]
  lay = layeredTransform(3,f)
  # check that various constructions are well-defined
  testTransform(f,lay)
  f = randn(1020)
  m=3
  lay = layeredTransform(m, f, subsampling=[8,4,2,1], nScales=[16,8,8,8], CWTType=WT.dog2, averagingLength=[16,4,4,2],averagingType=[:Mother for i=1:(m+1)],boundary=[WT.DEFAULT_BOUNDARY for i=1:(m+1)])
  testTransform(f, lay)
  layeredTransform(3, length(f), [2, 2, 2, 2], [2, 2, 2, 2], WT.dog2)
  layeredTransform(3, length(f), 8, [2, 2, 2, 2], WT.dog2)
  layeredTransform(3, length(f), [2, 2, 2, 2], 2, WT.dog2)
  layeredTransform(3, length(f), 8, 2, WT.dog2)

  # test on input with multiple samples
  testTransform(randn(2,3,1020),lay)
  testTransform(randn(2,1020),lay)
end
scattered(lay, randn(3,1020), "full")

# scattered construction tests
scattered{Float64,2}(2,1,[im*randn(10,10) for i=1:3],[randn(10,10) for i=1:3])
layers = layeredTransform(3, length(f), 8, 4)

results = scattered(layers, f)

f = randn(10,30,500) #there are 10 examples of length 500
layeredTransform(3,f[1,1,:])
#TODO: make a version that can handle input of arbitrary number of initial dimensions
tmp = layeredTransform(1,f[1,1,:])
ScatteringTransform.cwt(f, tmp.shears[1])
# averageNoiseMat = cwt(f,tmp.shears[1],J1=1)
# using FFTW
# A = randn(10,10,30)
# eachindex
# CI = CartesianIndex(size(A))
# for i in Tuple(CI)
#   print(i)
# end
# axes(A,3)
# for x in axes(A)[1:end-1]
#   print(x)
# end
# for x in eachindex.(axes(A)[1:end-1])
#   print("$(size(A[x]))")
# end
# A = randn(10,10,30)
# using LinearAlgebra
# for x in eachindex(view(A,axes(A)[1:end-1]..., 1))
#   print(x)
#   print("$(size(A[x,:]))")
#   println("     ")
# end
# CartesianIndex.(axes(A, 1), axes(A, 2))
# view(A,100,100, :)
# for i in eachindex(view(A,:,1))
#   @show i
# end
# randn(100,100,100).*
# using LinearAlgebra
#
# dw = zeros(10000,10000)
# v = rand(10000)
# h = rand(10000)
# @time for i = 1:10
#   dw += h*v'
# end
# @time for i = 1:10
#   BLAS.gemm!('N', 'T', 1.0, h, v, 1.0, dw)
# end

# TODO write tests for thinST
#      no outputSubsample
#      fixed size outputSubsample
#      fraction outputSubsample




# TODO: finish test folder transformations
# testing folder transformations
# using Distributed
# addprocs(8)
# @everywhere using Revise
# @everywhere using HDF5, JLD, ScatteringTransform
# using ScatteringTransform
# mkpath("tmpData")
# data = randn(20, 3, 5, 100)
# for i=1:20
#   mkpath("tmpData/tmp$(i)")
#   for j=1:3
#     h5write("tmpData/tmp$(i)/tmp$(j).h5", "data",data[i,j,:,:])
#   end
# end
# # direct transformation
# layers = layeredTransform(2,data)
# @time thinOutput = thinSt(data,layers,outputSubsample=(-1,3))
# # auxillary functions
# targetDir = "tmpOutput"
# function defaultLoadFunction(filename)
#   return (h5read(filename,"data"), true)
# end
# transformFolder("tmpData", targetDir, layers; separate=true, loadThis = defaultLoadFunction, postSubsample=(-1,3))
# load("tmpOutput/settings.jld","layers")
# load("/VastExpanse/data/SSAM2_motionCompensatedData/newData/lipschitz2Layer3Samples/settings.jld  ")
# layeredTransform
# @testset "folder transformations"
# end
# for i=1:20
#   mkpath("tmpData/tmp$(i)")
#   for j=1:3
#     h5write("tmpData/tmp$(i)/tmp$(j).h5", "data",data[i,j,:,:])
#   end
# end
# "α_"[1]
# layers
# using JLD
# module ScatteringTransform
# include("src/modifiedTransforms.jl")
# export layeredTransform
# struct layeredTransform{T}
#   m::Int # the number of layers, not counting the zeroth layer
#   n::Int # the length of a single entry
#   shears::Array{T} # the array of the transforms; the final of these is used only for averaging, so it has length m+1
#   subsampling::Array{Float64,1} # for each layer, the rate of subsampling. There is one of these for layer zero as well, since the output is subsampled, so it should have length m+1
#   layeredTransform{T}(m::Int, n::Int, shears::Array{T}, subsampling::Array{Float64,1}) where {T} = new(m,n, shears, subsampling)
# end
# end
# @everywhere
# ¥
# "¥_"[2]
