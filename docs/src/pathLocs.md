# Path Operations

To make working with paths somewhat easier, in addition to indexing `ScatteringOut` by layer and then raw index, there is the `pathLocs` type:

```jldoctest ex
julia> using ScatteringTransform, Wavelets

julia> St = scatteringTransform((1024,1,1),2)
stFlux{2, Nd=1, filters=[15, 14], σ = abs, batchSize = 1, normalize = true}

julia> s = St(testfunction(1024, "Doppler"))
ScatteredOut{Array{Float32},3} 1 dim. OutputSizes:
    (512, 1, 1)
    (342, 15, 1)
    (228, 14, 15, 1)

julia> p = pathLocs(2, (3,5))
pathLocs{3}((nothing, nothing, (Colon(), 3, 5, Colon())))

julia> s[p]
228×1 Matrix{Float32}:
 0.042854752
 0.05448776
 ⋮
 0.0009788324
 0.0010495521

```

`p` above for example, accesses the second layer path `(3,5)` (`3` being the second layer index and `5` being the first layer index).
The first entry specifies the layer, and the second specifies which entries in that layer.

```jldoctest ex
julia> p1 = pathLocs(2, (:,3))
pathLocs{3}((nothing, nothing, (Colon(), Colon(), 3, Colon())))

julia> s[p1]
228×14×1 Array{Float32, 3}:
[:, :, 1] =
 24.3279     0.0340069   …  0.000116684  0.000142872
 24.3125     0.0458974      0.000125332  0.00016506
  ⋮                      ⋱
  0.0150998  0.00992407     2.98423f-5   8.26744f-5
  0.0152391  0.0105978      2.61331f-5   7.9295f-5

```

`p1` grabs every path where the first layer index is `3`.

We can also grab multiple layers using a single path:

```jldoctest ex
julia> p2 = pathLocs(1, (5,), 2, (1,:))
pathLocs{3}((nothing, (Colon(), 5, Colon()), (Colon(), 1, Colon(), Colon())))

julia> s[p2][1]
342×1 Matrix{Float32}:
 0.0057057445
 0.007921134
 ⋮
 0.0002874717
 0.0003108125

julia> s[p2][2]
228×15×1 Array{Float32, 3}:
[:, :, 1] =
 0.0327152  8.44757    …  4.08167      3.19014
 0.038269   8.65702       4.06725      3.17879
 ⋮                     ⋱
 0.589429   0.0203722     0.000652602  0.000523428
 0.590318   0.02058       0.000660961  0.000529933

julia> p3 = pathLocs(1, s)
pathLocs{3}([(); (); … ; (); ();;;])
```

```@docs
pathLocs
nonZeroPaths
computeLoc
```
