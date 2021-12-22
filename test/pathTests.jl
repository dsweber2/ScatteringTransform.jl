@testset "Path Methods" begin
    t = 0.001:6π/100:6π
    f = 2 .* max.(0, -(t .- 3 * π)) ./ (t .- 3 * π) .* (sin.(2π * t)) .+ 10 .* max.(0, t .- 3 * π) ./ (t .- 3 * π) .+ max.(0, t .- 3 * π) ./ (t .- 3 * π) .* sin.(4π * t .+ π)
    layers = stParallel(2, length(f))
    output = st(f, layers, abs, thin = false)

    p = pathLocs(1, 1) # an object that we can use as an index into the output
    @test output[p] ≈ output[1][:, 1, 1]
    p = pathLocs(2, (1, 1))
    @test output[p] ≈ output[2][:, 1, 1, 1]
    p = pathLocs(2, (1, 2))
    @test output[p] ≈ output[2][:, 1, 2, 1]
    p = pathLocs(2, (2, 1))
    @test output[p] ≈ output[2][:, 2, 1, 1]
    p = pathLocs(2, (1, :))
    @test output[p] ≈ output[2][:, 1, :, 1]

    # path setindex
    Sx = ScatteredOut((randn(16, 1, 1), randn(11, 32, 1), randn(7, 27, 32, 1)))
    p = pathLocs(0, 5)
    newVal = randn()
    Sx[p] = newVal
    @test Sx[p][1] ≈ Sx.output[1][5, 1, 1]
    @test Sx[p][1] == newVal
    # setting a whole layer
    p = pathLocs(0, :)
    newVal = randn(16)
    Sx[p] = newVal
    @test Sx[p] ≈ Sx.output[1]
    @test Sx[p] ≈ newVal
    # setting a (later) whole layer
    p = pathLocs(2, :)
    newVal = randn(7, 27, 32, 1)
    Sx[p] = newVal
    @test Sx[p] ≈ Sx.output[3]
    @test Sx[p] ≈ newVal

    # setting just a couple of indices
    p = pathLocs(2, (1, 3))
    newVal = randn(7, 1)
    Sx[p] = newVal
    @test Sx[p] ≈ Sx.output[3][:, 1, 3, 1]
    @test Sx[p] ≈ newVal
end
