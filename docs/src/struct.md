# scatteringTransform type

```@docs
scatteringTransform
scatteringTransform(inputSize, m, backend::UnionAll; kwargs...)
stFlux
```

## Parameters unique to ScatteringTransform.jl

- `m::Integer=2`: the total number of layers
- `normalize::Bool=true`: if true, apply the function [`normalize`](@ref ScatteringTransform.normalize). The amount of energy in each layer decays exponentially for most inputs, so without normalizing, using the scattering coefficients poses some difficulty. To avoid this, when `normalize` is true, each layer is divided by the overall norm of that layer, and then multiplied by the number of paths in that layer.
- `poolBy::Union{<:Integer, <:Rational, <:Tuple}=3//2`: the amount to pool between layers. For a two layer network, the default is expanded to `(3//2, 3//2, 3//2)`, corresponding to the first layer, second layer, and final averaging subsampling rates. It also accepts tuples that are too short, and simply replicates the last entry, e.g. `poolBy=(2,3//2)` for a three layer network is equivalent to `poolBy=(2,3//2,3//2,3//2)`.
- `outputPool::Union{<:Integer, <:Rational, <:Tuple}=2`: the amount to subsample after averaging in each layer (present because averaging removes the high frequencies, so the full signal is inherently redundant). Has a similar structure to `poolBy`.
- `flatten::Bool=false`: if true, return a matrix that is `(nCoefficients,nExamples)`, otherwise, return the more structured [`ScatteredOut`](@ref ScatteringTransform.ScatteredOut)
- `σ::function=abs`: the nonlinearity applied pointwise between layers. This should take in a `Number`, and return a `Number` (real or complex floating point). If the wavelet transform is [analytic](https://dsweber2.github.io/ContinuousWavelets.jl/dev/coreType/#ContinuousWavelets.ContWave), this must have a method for `Complex`.

## Parameters passed to [FourierFilterFlux.jl](https://dsweber2.github.io/FourierFilterFlux.jl/dev/)

- `dtype::DataType=Float32`: the data type used to represent the filters.
- `cw::ContWaveClass=Morlet()`: the type of wavelet to use.
- `plan::Bool=true`: if true, store the fft plan for reuse.
- `convBoundary::ConvBoundary=Sym()`: the type of symmetry to use in computing the transform. Note that `convBoundary` and `boundary` are different, with `boundary` needing to be set using types from ContinuousWavelets and `convBoundary` needs to be set using the FourierFilterFlux types.
- `averagingLayer::Bool=false`: if true, use just the averaging filter, and drop all other filters.
- `trainable::Bool=false`: if true, the wavelet filters are considered parameters, and can be updated using standard methods from Flux.jl.
- `bias::Bool=false`: if true, include an offset, initialized using `init`. Most likely to be used with `trainable` true.
- `init::function=Flux.glorot_normal`: a function to initialize the bias, otherwise ignored.

## Parameters passed to [ContinuousWavelets.jl](https://dsweber2.github.io/ContinuousWavelets.jl/dev/)

+ `scalingFactor`, `s`, or `Q::Real=8.0`: the number of wavelets between the octaves ``2^J`` and ``2^{J+1}`` (defaults to 8, which is most appropriate for music and other audio). Valid range is ``(0,\infty)``.
+ `β::Real=4.0`: As using exactly `Q` wavelets per octave leads to excessively many low-frequency wavelets, `β` varies the number of wavelets per octave, with larger values of `β` corresponding to fewer low frequency wavelets(see [Wavelet Frequency Spacing](https://dsweber2.github.io/ContinuousWavelets.jl/dev/spacing/#Wavelet-Frequency-Spacing) for details).
  Valid range is ``(1,\infty)``, though around `β=6` the spacing is approximately linear *in frequency*, rather than log-frequency, and begins to become concave after that.
+ `boundary::WaveletBoundary=SymBoundary()`: The default boundary condition is `SymBoundary()`, implemented by appending a flipped version of the vector at the end to eliminate edge discontinuities. See [Boundary Conditions](https://dsweber2.github.io/ContinuousWavelets.jl/dev/bound/#Boundary-Conditions) for the other possibilities. 
+ `averagingType::Average=Father()`: determines whether or not to include the averaging function, and if so, what kind of averaging. The options are
  - `Father`: use the averaging function that corresponds to the mother Wavelet.
  - `Dirac`: use the sinc function with the appropriate width.
  - `NoAve`: don't average. this has one fewer filters than the other `averagingTypes`
+ `averagingLength::Int=4`:  the number of wavelet octaves that are covered by the averaging, 
+ `frameBound::Real=1`: gives the total norm of the whole collection, corresponding to the upper frame bound; if you don't want to impose a particular bound, set `frameBound<0`.
+ `normalization` or `p::Real=Inf`: the p-norm preserved as the scale changes, so if we're scaling by ``s``, `normalization` has value `p`, and the mother wavelet is ``\psi``, then the resulting wavelet is ``s^{1/p}\psi(^{t}/_{s})``.
  The default scaling, `Inf` gives all the same maximum value in the frequency domain.
  Valid range is ``(0,\infty]``, though ``p<1`` isn't actually preserving a norm.
