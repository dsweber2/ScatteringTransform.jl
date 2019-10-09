@everywhere function testFFTPlans(F::Future, dataS, isComplex, padBy)
    if isComplex || padBy[1] > 0
        P = fetch(F)
        if length(dataS)<2
            truthVal = (size(P) == dataS)
            if truthVal
                return true
            else
                println("$(size(P)) ?= $(dataS)")
            end
        else         # the above only happens for 1D data. The latter doesn't effect padding
            truthVal = (size(P) == ((dataS[1:2] .+ 2 .* padBy)..., dataS[3:end]...))
            if !truthVal
                println("$(size(P)) ?= $((dataS[1] + 2*padBy[1], dataS[2] + 2*padBy[2], dataS[3:end]...))")
            end
            return truthVal
        end
    else
        rP,P = fetch(F)
        t1 = (size(P) == dataS)
        t2 = (size(rP) == dataS)
        return min(t1, t2, eltype(rP) <: Real, eltype(P) <: Complex)
    end
end
@testset "plan pre-allocation" begin
    # 1D test
    depth=2
    dataS = (100, 1, 3)
    layers = layeredTransform(depth, 100)
    n, q, dataSizes, outputSizes, resultingSize =
        ScatteringTransform.calculateSizes(layers, (-1,-1),
                                           dataS)
    outputSizes
    # there's only one plan when it's complex
    plans = createFFTPlans(layers, dataSizes, iscomplex=true)
    @test size(plans) == (nworkers(), depth+1)

    # test that the sizes fit for mirroring the data
    mirSize = [(2dS[1], dS[3:end]...) for dS in dataSizes]
    @test minimum(remotecall_fetch(testFFTPlans, i, plans[i,j], mirSize[j], true, (0,0)) for i=1:size(plans,1) for j=1:size(plans,2))


    plans = createFFTPlans(layers, dataSizes, iscomplex=false)
    @test minimum(remotecall_fetch(testFFTPlans, i, plans[i,j], mirSize[j], false, (0,0)) for i=1:size(plans,1) for j=1:size(plans,2))
end
