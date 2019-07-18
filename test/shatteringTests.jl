######################################################################################################
###################################### Shattering Tests ##############################################
######################################################################################################
function testLayerConstruction(layer::layeredTransform, X::Array{<:Real,N},
                               m::Int64, xSubsampled::Array{Int64},
                               ySubsampled::Array{Int64},
                               subsampling::Vector{<:Real}) where {N}
    @test layer.m==m
    @test layer.subsampling == subsampling
    @test sizes(bsplineType(), layer.subsampling, size(X)[end-1]) == xSubsampled
    @test sizes(bsplineType(), layer.subsampling, size(X)[end]) == ySubsampled
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
m3=2
X3 = zeros(Float32, 1, 50, 50)
X3[1, 13:37, 13:37] = ones(Float32, 25, 25)
layer3 = layeredTransform(m3, size(X3)[end-1:end], typeBecomes=eltype(

    X3))
asdf = st(X3, layer3, absType())
results = ScatteringTransform.wrap(layer3, asdf, X3)
@testset "layer construction" begin
    xSubsampled = [50, 25, 13, 7]; ySubsampled = [55, 28, 14, 7]
    testLayerConstruction(layer1, X1, m1, xSubsampled, ySubsampled, [2.0 for
                                                                     i=1:m1+1])
    xSubsampled = [201, 134, 90, 60]; ySubsampled= [325, 217, 145, 97]
    testLayerConstruction(layer2, X2, m2, xSubsampled, ySubsampled, [subsamp
                                                                     for i =
                                                                     1:m2+1])
    xSubsampled = [50, 25, 13, 7]; ySubsampled = [50, 25, 13, 7]
    testLayerConstruction(layer3, X3, m3,xSubsampled, ySubsampled, [2.0 for i=1:m3+1])
end


################# Subsampling methods #################
function testSubsampling(X::Array{T},layer::layeredTransform,m::Int64) where T<:Real
    tmpX = Array{T}(X)
    tmpSizesx = sizes(bsplineType(), layer.subsampling,size(X)[end-1])
    tmpSizesy = sizes(bsplineType(), layer.subsampling,size(X)[end])
    for i=1:m1+1
        @test size(tmpX)[end-1:end] == (tmpSizesx[i],tmpSizesy[i])
        tmpX = resample(tmpX, layer.subsampling[i])
    end
end
@testset "sizes in the layered transform match those produced by subsampling" begin
    testSubsampling(X1, layer1, m1)
    testSubsampling(X2, layer2, m2)
    testSubsampling(X3, layer3, m3)
end

#TODO: test different percentages
function testShattered(X::Array{T,N},layer::layeredTransform,m::Int64) where {T <:Real,N}
    thing = scattered(layer, X)
    @test typeof(thing.data[1]) == Array{T, N+1}
    @test typeof(thing.output[1]) == Array{T, N+1}
    @test thing.m ==layer.m
    # test the subsampling of the array is correct
    n, q, dataSizes, outputSizes, resultingSize =
        ScatteringTransform.calculateSizes(layer, (-1,-1), size(X))
    @test [size(thing.data[i])[end-2] for i=1:m+1] == sizes(bsplineType(),
                                                       layer.subsampling,
                                                       size(X)[end-1])[1:3]
    @test [size(thing.data[i])[end-1] for i=1:m+1] == sizes(bsplineType(),
                                                       layer.subsampling,
                                                       size(X)[end])[1:3]
    @test [size(thing.output[i])[end-2] for i=1:m+1] == [outputSizes[i][end-2] for
                                                    i=1:m+1]
    @test [size(thing.output[i])[end-1] for i=1:m+1] == [outputSizes[i][end-1] for
                                                   i=1:m+1]
    # check the number of scales
    @test [size(thing.data[i])[end] for i=1:m+1] == [prod([1; q][1:i]) for i=1:m+1]
    @test [size(thing.output[i])[end] for i=1:m+1] == [prod([1; q][1:i]) for i=1:m+1]
end



function testShattering(X::Array{Float64}, nonlinear::S, layer::layeredTransform, m::Int64) where S <: nonlinearity
    @time outputFull = st(X,layer,nonlinear,thin=false)

    # check that the data isn't NaN
    @test minimum([isfinite(maximum(abs.(outputFull.data[i]))) for i=1:m+1])
    @test minimum([isfinite(maximum(abs.(outputFull.output[i]))) for i=1:m+1])

    # check that the data isn't actually zero when the input is nonzero TODO:
    # when data is functional
    @test minimum(minimum(abs.(outputFull.output[i])) for i=1:m+1) != 0.0

    # see how the thin version is doing
    @time outputThin = st(X,layer,nonlinear,thin=true)
    @test isfinite(maximum(abs.(outputThin)))
    @test minimum(minimum(abs.(outputFull.output[i])) for i=1:m+1) != 0.0

    # compare the two
    @test maximum(abs.(flatten(outputFull, layer)-outputThin))==0.0
    return outputFull
end


@testset "shattered construction" begin
    # type input works properly
    testShattered(X1, layer1, m1)
    testShattered(X2, layer2, m2)
    testShattered(X3, layer3, m3)
end

# test the nonlinearity definitions
t = [x+im*y for x=-10:1/5:10, y=-10:1/5:10]
@testset "nonlinearity definitions" begin
    @test ScatteringTransform.abs.(t)==abs.(t)
    @test ScatteringTransform.ReLU.(t) == max.(0,real(t))+im*max.(0,imag(t))
    println("halfway through")
    @test ScatteringTransform.Tanh.(t) == tanh.(real(t))+im*tanh.(imag(t))
    @test ScatteringTransform.softplus.(t) == ((log.(1 .+ exp.(real.(t))) .- log(2)) + im .*(log.(1
                                                                              .+
                                                                              exp.(imag.(t)))
                                                                         .-log(2)))
end


function testShattering(X::Array{Float64}, nonlinear::S, layer::layeredTransform, m::Int64) where S <: nonlinearity
    @time outputFull = st(X,layer,nonlinear,thin=false)

    # check that the data isn't NaN
    @test minimum([isfinite(maximum(abs.(outputFull.data[i]))) for i=1:m+1])
    @test minimum([isfinite(maximum(abs.(outputFull.output[i]))) for i=1:m+1])

    # check that the data isn't actually zero when the input is nonzero TODO:
    # when data is functional, also make sure that's non-zero
    @test minimum(minimum(abs.(outputFull.output[i])) for i=1:m+1) != 0.0

    # see how the thin version is doing
    @time outputThin = st(X,layer,nonlinear,thin=true)
    @test isfinite(maximum(abs.(outputThin)))
    @test minimum(minimum(abs.(outputFull.output[i])) for i=1:m+1) != 0.0

    # compare the two
    @test maximum(abs.(flatten(outputFull, layer)-outputThin))==0.0
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
