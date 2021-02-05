@testset "Path Methods" begin
    t = 0.001:6π/100:6π
    f = 2 .* max.(0,-(t.-3*π))./(t.-3*π) .* (sin.(2π*t)) .+ 10 .* max.(0,t.-3*π)./(t.-3*π) .+ max.(0,t.-3*π)./(t.-3*π).* sin.(4π*t.+π)
    layers = stParallel(2, length(f))
    output = st(f, layers, abs, thin=false)

    p = pathLocs(1,1) # an object that we can use as an index into the output
    @test output[p] ≈ output[1][:,1,1]
    p = pathLocs(2,(1,1))
    @test output[p] ≈ output[2][:,1,1,1]
    p = pathLocs(2, (1,2))
    @test output[p] ≈ output[2][:,1,2,1]
    p = pathLocs(2, (2,1))
    @test output[p] ≈ output[2][:,2,1,1]
    p = pathLocs(2, (1,:))
    @test output[p] ≈ output[2][:,1,1:13,1]
end
