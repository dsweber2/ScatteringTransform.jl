using Test
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
