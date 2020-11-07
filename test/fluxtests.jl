@testset "Flux Scattering Transform methods"
ifGpu = identity
i=40; s =6//5
nExtraDims = 2
xExtraDims = 3
k = 4; N=2
i=25; s = subsampRates[1]; xExtraDims= 2; k =windowSize[1]; (N,nExtraDims) = NdimsExtraDims[1]
import ScatteringTransform:stopAtExactly_WithRate_
@testset "testing pooling" begin
    subsampRates = [3//2, 2, 5//2, 6//5]
    windowSize=[2,3,4]
    NdimsExtraDims = [(N,extra) for extra in 2:4, N in 1:2 if !(N==2 && extra==4)]
    @testset "bizzare pooling sizes 1D i=$i, s=$s, nExtraDims=$nExtraDims, xExtraDims=$xExtraDims, k=$k N=$N" for i=25:3:40, s in subsampRates, xExtraDims in 2:nExtraDims, k in windowSize, (N,nExtraDims) in NdimsExtraDims
        selectXDims = ntuple(x->2, xExtraDims)
        x = ifGpu(randn(ntuple(x->i, N)...,rand(2:10, xExtraDims)...)); size(x)
        r = RationPool(ntuple(x->s, N),k,nExtraDims=nExtraDims)
        @test length(r.m.k) == length(r.resSize)+nExtraDims-2
        nPoolDims(r)
        Nneed = ndims(r.m)+2
        Sx = r(x);
        @test size(Sx)==(poolSize(r, size(x))..., size(x)[N+1:end]...)
        # does it take the mean we expect?
        chosenLocs = stopAtExactly_WithRate_(i-k+1,s)
        loc = chosenLocs[5]
        neighborhood = 0:k-1
        neighborhood = ntuple(x->loc .+ neighborhood, N)
        @test Sx[ntuple(x->5, N)...,selectXDims...] ≈ 
            mean(x[neighborhood...,selectXDims...])
        # which entries matter for the 5th entry, according to the gradient?
        ∇ = gradient(x->r(x)[ntuple(x->5, N)..., selectXDims...],x);
        δ = zeros(size(x)); δ[neighborhood..., selectXDims...] .= 1/k^N
        @test findall(∇[1].!=0) == findall(δ.!=0)
        @test ∇[1] ≈ δ
    end
end

@test ScatteringTransform.nPathDims(1)==1
@test ScatteringTransform.nPathDims(2)==1
@test ScatteringTransform.nPathDims(3)==2

@testset "2D basics" begin
    init = ifGpu(10 .+ randn(32,32, 3, 2));
    sst = stFlux(size(init), 2, poolBy=3//2)
    res = sst(init);
    @test length(res.output)== 2+1
    @test size(res.output[1]) == (16, 16, 3, 2)
    @test minimum(abs.(res.output[1])) > 0
    @test size(res.output[2]) == (10, 10, 144, 2)
    @test minimum(abs.(res.output[2])) > 0
    @test size(res.output[3]) == (6, 6, 16, 144, 2)
    @test minimum(abs.(res.output[3])) > 0
    totalSize = 16*16*3 + 10*10*144 + 6*6*16*144
    smooshed = ScatteringTransform.flatten(res);
    @test size(smooshed) ==(totalSize, 2)
end

@testset "1D basics" begin
    init = ifGpu(10 .+ randn(64, 3, 2));
    sst = stFlux(size(init), 2, poolBy=3//2, outputPool=(2,))
    res = sst(init)
    @test length(res.output)== 2+1
    @test size(res.output[1]) == (32, 3, 2)
    @test minimum(abs.(res.output[1])) > 0
    @test size(res.output[2]) == (20, 3*11, 2)
    @test minimum(abs.(res.output[2])) > 0
    @test size(res.output[3]) == (13, 8, 3*11, 2)
    @test minimum(abs.(res.output[3])) > 0
    totalSize = 32*3 + 20*3*11 + 13*8*3*11
    smooshed = ScatteringTransform.flatten(res);
    @test size(smooshed) == (totalSize, 2)
end

# integer pooling rate
@testset "2D integer pooling" begin
    stEx = stFlux((131,131,1,1), 2, poolBy=3)
    stEx.outputSizes
    c = stEx.mainChain
    size(c[3](c[2](c[1](randn(131,131,1,1)))))
    scat = stEx(ifGpu(randn(131,131,1,1)))
    @test size(stEx.mainChain[1].fftPlan) == (391, 391, 1, 1)
    @test size(stEx.mainChain[4].fftPlan) == (96, 106, 48, 1)
    @test size(stEx.mainChain[7].fftPlan) == (43,43,48,48,1)
    @test stEx.mainChain[1].bc.padBy == (130,130)
    @test stEx.mainChain[4].bc.padBy == (26, 31)
    @test stEx.mainChain[7].bc.padBy == (14, 14)
    @test stEx.outputPool == ntuple(i->(2,2),3)
    @test ndims(stEx)==2
    resultSize = ((66, 66, 1, 1), (22, 22, 48, 1), (8, 8, 48, 48, 1))
    @test stEx.outputSizes == resultSize
    @test ([size(s) for s in scat.output]...,) == stEx.outputSizes 
end

@testset "1D integer pooling" begin
    stEx = stFlux((131,1,1), 2, poolBy=3)

    scat = stEx(ifGpu(randn(131,1,1)));
    @test size(stEx.mainChain[1].fftPlan[1]) == (2*131,1,1)
    @test size(stEx.mainChain[1].fftPlan[2]) == (2*131,1,1)
    @test size(stEx.mainChain[4].fftPlan[1]) ==(2*44, 14,1)
    @test size(stEx.mainChain[4].fftPlan[2]) ==(2*44, 14,1)
    @test size(stEx.mainChain[7].fftPlan[1]) == (2*15,8,14,1)
    @test size(stEx.mainChain[7].fftPlan[2]) == (2*15,8,14,1)
    @test stEx.mainChain[1].bc == Sym()
    @test stEx.mainChain[4].bc == Sym()
    @test stEx.mainChain[7].bc == Sym()
    @test stEx.outputPool == ((2,), (2,), (2,))
    @test ndims(stEx)==1

    resultSize = ((66, 1, 1), (22, 14, 1), (8, 8, 14, 1))
    @test stEx.outputSizes == resultSize
    @test ([size(s) for s in scat.output]...,) == resultSize
    @test stEx.outputSizes == ([size(s) for s in scat.output]...,)
end

# rolling and flattening does nothing
@testset "roll and flatten 2D" begin
    init = ifGpu(randn(32,32, 1, 2));
    sst = stFlux(size(init), 2, poolBy=3//2)
    res = sst(init);
    smooshed = ScatteringTransform.flatten(res);
    if ifGpu!= identity
        @test typeof(smooshed) <: CuArray
    end

    reconst = roll(smooshed, sst);
    @test all(reconst .≈ res)
    @test typeof(reconst.output) <: Tuple
end

@testset "roll and flatten 1D" begin
    init = ifGpu(randn(64, 1, 2));
    sst = stFlux(size(init), 2, poolBy=3//2)
    res = sst(init);
    smooshed = ScatteringTransform.flatten(res);
    if ifGpu!= identity
        @test typeof(smooshed) <: CuArray
    end
    reconst = roll(smooshed, sst);
    @test all(reconst .≈ res)
    @test typeof(reconst.output) <: Tuple
end

# normalization in 2D
x = randn(10,4,3,5,7);
Nd=2;
xp = ScatteringTransform.normalize(x,2);
for w in eachslice(xp,dims=ndims(x))
    @test norm(w,2) ≈ 3*5
end

# normalization in 1D
x = randn(10,3,5,7);
xp = ScatteringTransform.normalize(x,1);
for w in eachslice(xp,dims=ndims(x))
    @test norm(w,2) ≈ 3*5
end
end
@testset "pathLocs" begin
    using ScatteringTransform:parseOne
    exs = Colon()
    # layer 0
    @test (34:53, 1, :)==parseOne((0,34:53),1, exs)
    @test (34:53, 34:53, 1, :)==parseOne((0,(34:53, 34:53)),2,exs)
    @test (:,:,:)==parseOne((0,:),1,exs)
    @test (:,:,:,:)==parseOne((0,:),2,exs)
    @test (4:35,1,1:10) == parseOne((0,(4:35,1,1:10)),1,exs)
    @test (4:35,1,1,1:10) == parseOne((0,(4:35,1,1,1:10)),2,exs)
    # layer 1
    @test (:, 34:53, :) == parseOne((1,34:53),1,exs)
    @test (:, 5, :) == parseOne((1,5),1,exs)
    @test (:, :, 34:53, :)==parseOne((1,34:53),2,exs)
    @test (:,:, 5, :) == parseOne((1,5),2,exs)
    @test (:,:,:)==parseOne((1,:),1,exs)
    @test (:,:,:,:)==parseOne((1,:),2,exs)
    @test (4:35,1,1:10) == parseOne((1,(4:35,1,1:10)),1,exs)
    @test (4:35,1,1,1:10) == parseOne((1,(4:35,1,1,1:10)),2,exs)
    # layer 2
    @test (:, :, 34:53, :) == parseOne((2,34:53),1,exs)
    @test (:, :, :, 34:53, :)==parseOne((2,34:53),2,exs)
    @test (:, 34:53, 5, :) == parseOne((2,(34:53,5)),1,exs)
    @test (:, :, 34:53, 5, :)==parseOne((2,(34:53,5)),2,exs)
    @test (:,:,:,:)==parseOne((2,:),1,exs)
    @test (:,:,:,:,:)==parseOne((2,:),2,exs)
    @test (4:35,1,1:10,1,:) == parseOne((2,(4:35,1,1:10)),1,exs)
    @test (4:35,1,1,1:10,1,:) == parseOne((2,(4:35,1,1,1:10)),2,exs)

    # actual pathLocs construction
    @test ((:,:,:),(:,3:4,:), (:,5,:,:))==pathLocs(1,3:4, 0,:, 2,(5,:)).indices
    @test ((:,:,:),nothing, (:,5,:,:))==pathLocs(0,:, 2,(5,:)).indices

    # pathLocs usage
    ex = ScatteredOut((randn(50,1,1), randn(34,13,1), randn(23,11,13,1)))
    p = pathLocs(0,:, 2,(5,:))
    @test minimum((ex.output[1], ex.output[3][:,5,:,:]) .≈ ex[p])
    p = pathLocs(2, (4:9, 5))
    @test ex.output[3][:,4:9,5,:]≈ ex[p]
    p = pathLocs(1,:)
    @test ex.output[2] ≈ ex[p]
    p = pathLocs()
    @test minimum(ex[p] .≈ ex.output)

    # setindex (some more in the nonZero paths section)
    p = pathLocs(2, (4:9, 5))
    newVal = randn(23,6,1) 
    ex[p] = newVal
    @test ex[p] ≈ newVal #TODO broken
    
    p = pathLocs(2, (4:9, 5), 1, (3:5,))
    newVal = (randn(34,3,1), randn(23,6,1))
    ex[p] = newVal
    @test all(ex[p] .≈ newVal)

    # this is still broken; some sort of broadcast shenanigans
    p = pathLocs(0,40:46)
    ex[p] .= 1
    @test minimum(ex[p] .≈ 1) #TODO broken
    
    # single entries behave a bit strangely
    p = pathLocs(0, 3)
    newVal = randn()
    ex[p] = newVal
    @test minimum(ex[p] .≈ newVal)
    

    # getindex using ints and arrays
    @test ex[0] == ex.output[1]
    @test ex[1] == ex.output[2]
    @test ex[0:1] == ex.output[1:2]

    # pathLoc getindex adjoint
    p = pathLocs(2, (4:9, 5))
    y, back = pullback(x->x[p], ex)
    res = back(y)
    res[1][p]
    ex[p]
    @test res[1][p] == ex[p] # TODO this is broken because an assignment above is broken
    anti_p = pathLocs(2, ([1:3..., 10,11], [1:4... 6:13...]))
    @test res[1][anti_p] ==zeros(23,5,1,12,1)
    @test maximum(maximum.(res[1][0:1])) == 0

    y, back = pullback(getindex, ex, 1);
    res = back(y);
    @test res[1][1] == ex[1]
    @test all(res[1][0] .== 0)
    @test all(res[1][2][1] .== 0)
    @test res[2]==nothing

    y, back = pullback(getindex, ex, 1:2);
    res = back(y);
    @test res[1][1:2] == ex[1:2]
    @test all(res[1][0] .== 0)
    @test res[2]==nothing

    # catting paths (may result in extra indices grabbed)
    pathsByHand = (pathLocs(0,40:46,exs=1), pathLocs(1,(24:26,4),exs=2),
                   pathLocs(1,(29:31,6),exs=1:2),
                   pathLocs(2,(11:14,10,1), exs=1), 
                   pathLocs(2, (11:15, 3, 5:7), exs=1))
    joint = cat(pathsByHand...)
    
    # nonZeroPaths
    ex = ScatteredOut((zeros(50,1,2), zeros(34,13,2), zeros(23,11,13,2)))
    for (ii, p) in enumerate(pathsByHand)
        ex[p] = (-1)^ii * ii * ones(size(ex[p]))
    end
    paths = nonZeroPaths(ex,wholePath=false, allTogetherInOne=true)
    @test ex[paths][1] == ex[pathLocs(0,40:46,exs=1)]
    @test ex[paths][2] == [fill(-3.0,3)...; fill(2.0,3)...; fill(-3.0,3)...] #not
    #easy to describe the location
    @test findall(paths.indices[2]) == [CartesianIndex(29, 6, 1),CartesianIndex(30, 6, 1),CartesianIndex(31, 6, 1),CartesianIndex(24, 4, 2),CartesianIndex(25, 4, 2),CartesianIndex(26, 4, 2),CartesianIndex(29, 6, 2),CartesianIndex(30, 6, 2),CartesianIndex(31, 6, 2)]
    @test findall(paths.indices[3]) == [CartesianIndex(11, 10, 1, 1),CartesianIndex(12, 10, 1, 1),CartesianIndex(13, 10, 1, 1),CartesianIndex(14, 10, 1, 1),CartesianIndex(11, 3, 5, 1),CartesianIndex(12, 3, 5, 1),CartesianIndex(13, 3, 5, 1),CartesianIndex(14, 3, 5, 1),CartesianIndex(15, 3, 5, 1),CartesianIndex(11, 3, 6, 1),CartesianIndex(12, 3, 6, 1),CartesianIndex(13, 3, 6, 1),CartesianIndex(14, 3, 6, 1),CartesianIndex(15, 3, 6, 1),CartesianIndex(11, 3, 7, 1),CartesianIndex(12, 3, 7, 1),CartesianIndex(13, 3, 7, 1),CartesianIndex(14, 3, 7, 1),CartesianIndex(15, 3, 7, 1)]
    @test ex[paths][3] == [fill(4.0, 4)...; fill(-5.0, 15)...]
    
    # setindex using a boolean array instead of a normal one
    mostlyNull = ScatteredOut((zeros(50,1,2), zeros(34,13,2), zeros(23,11,13,2)))
    newValues = randn(35)
    mostlyNull[paths] #todo this should be a tuple not an array
    mostlyNull[paths] = newValues
    @test cat(mostlyNull[paths]..., dims=1) ≈ newValues
    # testing adding one location at a time
    nullEx = ScatteredOut((zeros(50,1,1), zeros(34,13,1), zeros(23,11,13,1)))
    nullEx[0][3,1,1] = 50
    nullEx[2][3:20,1,1,1] .= 25; nullEx[2][5,2:5,1,1] .= 26; nullEx[2][5,9,12:13,1] .= 2
    addFrom = nonZeroPaths(nullEx, allTogetherInOne=true,wholePath=false)
    addTo = addNextPath(addFrom)
    @test findfirst(addTo.indices[1]!=addFrom.indices[1])==nothing
    @test addTo.indices[2] == nothing
    @test findfirst(addTo.indices[3])==nothing

    addToNext = addNextPath(addTo, addFrom)
    @test addToNext.indices[2] == nothing
    @test findfirst(addToNext.indices[1]!=addFrom.indices[1])==nothing
    @test findfirst(addToNext.indices[3].!=addFrom.indices[3])==CartesianIndex(4,1,1,1)
    @test findall(addToNext.indices[3]) == findall(addFrom.indices[3])[1:1]
    ii=2
    while addTo!=addToNext && ii <=24
        global addToNext, addTo,ii
        addTo = addToNext
        addToNext = addNextPath(addTo, addFrom)
        @test findall(addToNext.indices[3]) == findall(addFrom.indices[3])[1:ii]
        ii+=1
    end
    @test all(addToNext.indices .== addFrom.indices)
end
end
