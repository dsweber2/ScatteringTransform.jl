@testset "Path Methods" begin
    t = 0.001:6π/100:6π
    f = 2 .* max.(0,-(t.-3*π))./(t.-3*π) .* (sin.(2π*t)) .+ 10 .* max.(0,t.-3*π)./(t.-3*π) .+ max.(0,t.-3*π)./(t.-3*π).* sin.(4π*t.+π)
    layers = layeredTransform(2, length(f))
    output = st(f, layers, absType(), thin=false)

    p = pathType([1]) # an object that we can use as an index into the output
    @test output[p] ≈ output[2][:,1,1]
    p = pathType([1,1])
    @test output[p] ≈ output[3][:,1,1]
    p = pathType([1,2])
    @test output[p] ≈ output[3][:,2,1]
    p = pathType([2,1])
    @test output[p] ≈ output[3][:,1 + 9,1]
    p = pathType([1,:])
    @test output[p] ≈ output[3][:,1:9,1]
end
