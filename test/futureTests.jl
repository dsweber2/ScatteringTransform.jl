########################### Parallel FFT tests ################################
# using Distributed
# TODO: make a functional test here
# TODO: define a simple way of establishing Scattering transform on n workers
# addprocs(2)
# @everywhere using Revise
# @everywhere using ScatteringTransform
# m1=2; n1=50; p1=55
# X1 = randn(n1,p1)
# layer1 = layeredTransform(m1, size(X1), typeBecomes=eltype(X1))
# n, q, dataSizes, outputSizes, resultingSize =
#     ScatteringTransform.calculateSizes(layer1, (-1,-1), size(X1))

# plans = createFFTPlans(layer1, dataSizes, verbose = true, T=eltype(X1))
# fetch(plans[1,1])
# layers = layer1
# padBy = getPadBy(layers.shears[1])
# thing = createRemoteFFTPlan(dataSizes[1], padBy, Float64,false); size(thing)
# ptrThing = remotecall(createRemoteFFTPlan, 2, dataSizes[1][(end-2):(end-1)], padBy, Float64, false)
# tmp = fetch(ptrThing); typeof(tmp)
# FFTs = Array{Future,2}(undef, nworkers(), layers.m+1)




















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









#TODO fix the 1D tests
# function testConstruction1D(lay::layeredTransform, m::Int, shearType, averagingLength, averagingType, nShears::Vector{Int}, subsampling::Array{Float64,1}, n::Int)
#   @test lay.m == m
#   @test lay.subsampling==subsampling
#   sizes = ScatteringTransform.sizes(bsplineType(), bspline,lay.subsampling,n)
#   for i=1:length(lay.shears)
#     @test typeof(lay.shears[i]) == shearType[i]
#     @test lay.shears[i].averagingLength == averagingLength[i]
#     @test lay.shears[i].averagingType == averagingType[i]
#     @test numScales(lay.shears[i], sizes[i]) == nShears[i]
#   end
# end
# n = 10214
# f = randn(n)
# lay = layeredTransform(1, size(f)[end])
# m=3
# lay = layeredTransform(m, f, subsampling=[8,4,2,1], nScales=[16,8,8,8], CWTType=WT.dog2, averagingLength=[16,4,4,2],averagingType=[:Mother for i=1:(m+1)],boundary=[WT.DEFAULT_BOUNDARY for i=1:(m+1)])
# @testset "construction tests" begin
#   testConstruction1D(lay, 3, [ScatteringTransform.CFWA{Wavelets.WT.PerBoundary} for i=1:4], [16, 4, 4, 2], [:Mother for i=1:4], [165,62,46,40], [8,4,2,1]*1.0, n)
# end
# function testTransform(f::Array{T}, lay::layeredTransform) where T<:Number
#   n=size(f)[end]
#   @time output = st(f,lay)
#   sizes = ScatteringTransform.sizes(bsplineType(), bspline,lay.subsampling,n)
#   actualSizes = fill(0, size(sizes))
#   actualSizes[1] = size(output.data[1])[end-1]
#   for i=1:length(actualSizes)-2
#     @test size(output.data[i+1])[end-1] == size(output.output[i])[end-1]
#     actualSizes[i+1] = size(output.data[i+1])[end-1]
#   end
#   actualSizes[end] = size(output.output[end])[end-1]
#   @test actualSizes== sizes
#   # check the number of paths grows correctly
#   effectiveSize = length(size(f))==1 ? (1,size(f)[1]) : size(f)
#   numTransformed = ScatteringTransform.getQ(lay, effectiveSize)
#   @test numTransformed[2:end]==([size(out)[end] for out in output.output])[2:end]
#   @test eltype(output.data) == Array{Complex{eltype(f)},1+length(effectiveSize)}
#   @test eltype(output.output) == Array{eltype(f),1+length(effectiveSize)}
# end
# @testset "transform tests" begin
#   f=1.0 .*[1:50; 50:-1:1]
#   lay = layeredTransform(3,f)
#   # check that various constructions are well-defined
#   testTransform(f,lay)
#   f = randn(1020)
#   m=3
#   lay = layeredTransform(m, f, subsampling=[8,4,2,1], nScales=[16,8,8,8], CWTType=WT.dog2, averagingLength=[16,4,4,2],averagingType=[:Mother for i=1:(m+1)],boundary=[WT.DEFAULT_BOUNDARY for i=1:(m+1)])
#   testTransform(f, lay)
#   layeredTransform(3, length(f), [2, 2, 2, 2], [2, 2, 2, 2], WT.dog2)
#   layeredTransform(3, length(f), 8, [2, 2, 2, 2], WT.dog2)
#   layeredTransform(3, length(f), [2, 2, 2, 2], 2, WT.dog2)
#   layeredTransform(3, length(f), 8, 2, WT.dog2)

#   # test on input with multiple samples
#   testTransform(randn(2,3,1020),lay)
#   testTransform(randn(2,1020),lay)
# end
# scattered(lay, randn(3,1020), "full")

# # scattered construction tests
# scattered{Float64,2}(2,1,[im*randn(10,10) for i=1:3],[randn(10,10) for i=1:3])
# layers = layeredTransform(3, length(f), 8, 4)

# results = scattered(layers, f)

# f = randn(10,30,500) #there are 10 examples of length 500
# layeredTransform(3,f[1,1,:])
# #TODO: make a version that can handle input of arbitrary number of initial dimensions
# tmp = layeredTransform(1,f[1,1,:])
# ScatteringTransform.cwt(f, tmp.shears[1])
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
