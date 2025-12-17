# Path Operations

To make working with paths somewhat easier, in addition to indexing `ScatteringOut` by layer and then raw index, there is the `pathLocs` type:

```jldoctest ex
julia> using ScatteringTransform, Wavelets, Logging

julia> St = with_logger(NullLogger()) do
           scatteringTransform((1024,1,1),2)
       end
stFlux{Nd=1, m=2, filters=[15, 14], σ = abs, batchSize = 1, normalize = true}

julia> s = St(testfunction(1024, "Doppler"))
ScatteredOut{Array{Float32},3} 1 dim. OutputSizes:
    (512, 1, 1)
    (342, 15, 1)
    (228, 14, 15, 1)

julia> p = pathLocs(2, (3,5))
pathLocs{3}((nothing, nothing, (Colon(), 3, 5, Colon())))

julia> s[p]
228×1 Matrix{Float32}:
 0.09841786
 0.10313512
 ⋮
 0.00073056365
 0.00075656467
```

`p` above for example, accesses the second layer path `(3,5)` (`3` being the second layer index and `5` being the first layer index).
The first entry specifies the layer, and the second specifies which entries in that layer.

```jldoctest ex
julia> p1 = pathLocs(2, (:,3))
pathLocs{3}((nothing, nothing, (Colon(), Colon(), 3, Colon())))

julia> s[p1]
228×14×1 Array{Float32, 3}:
[:, :, 1] =
 24.4596     0.121606    …  0.000137235  0.000253008
 24.442      0.133231       0.000137483  0.000266736
  ⋮                      ⋱
  0.0143983  0.00752869     3.26755f-5   7.5444f-5
  0.0145095  0.00778378     3.2883f-5    7.69233f-5
```

`p1` grabs every path where the first layer index is `3`.

We can also grab multiple layers using a single path:

```jldoctest ex
julia> p2 = pathLocs(1, (5,), 2, (1,:))
pathLocs{3}((nothing, (Colon(), 5, Colon()), (Colon(), 1, Colon(), Colon())))

julia> s[p2][1]
342×1 Matrix{Float32}:
 0.10331459
 0.10902256
 ⋮
 0.00016353851
 0.00016699042

julia> s[p2][2]
228×15×1 Array{Float32, 3}:
[:, :, 1] =
 0.0620106  9.75739    …  4.02497      3.14531
 0.0652926  9.90173       4.0116       3.1348
 ⋮                     ⋱
 0.0193751  0.589259      0.000606986  0.000488191
 0.0195244  0.590126      0.000613764  0.000493479

julia> p3 = pathLocs(1, s)
pathLocs{3}([(); (); … ; (); ();;;])
```

```@docs
pathLocs
nonZeroPaths
computeLoc
```
