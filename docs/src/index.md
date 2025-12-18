# ScatteringTransform.jl

A julia implementation of the scattering transform, which provides a prestructured alternative to a convolutional neural network.
In a similar vein to a CNN, it alternates between continuous wavelet transforms, nonlinear function applications, and subsampling.
This library is end-to-end differentiable and runs on the GPU; there is a companion package, [ParallelScattering.jl](https://github.com/dsweber2/ParallelScattering.jl/) that runs on parallelized CPUs.

This is achieved by creating differentiable wavelet Fourier filters using [FourierFilterFlux](https://dsweber2.github.io/FourierFilterFlux.jl/dev/), which are then interspersed with [Subsampling Operators](@ref) modified from Flux.jl, and pointwise nonlinear functions (in practice, this means absolute value or ReLU).

For a comparable package in python, see [Kymatio](https://www.kymat.io/).

```@contents

```

## Basic Example

```@setup
using ScatteringTransform, ScatteringPlots, Wavelets, Plots
```

As an example signal, lets work with the doppler signal:

```@example ex
using Wavelets, Plots
N = 2047
f = testfunction(N, "Doppler")
plot(f, legend=false, title="Doppler signal")
savefig("rawDoppler.svg"); #hide
```

![](rawDoppler.svg)

First we need to make a `scatteringTransform` instance, which will create and store all of the necessary filters, subsampling operators, nonlinear functions, etc.
The parameters are described in the [scatteringTransform type](@ref).
Since the Doppler signal is smooth, but with varying frequency, let's set the wavelet family `cw=Morlet(π)` specifies the mother wavelet to be a Morlet wavelet with mean frequency π, and frequency spacing `β=2`:

```@example ex
using ScatteringTransform, ContinuousWavelets
St = scatteringTransform((N, 1, 1), 2, cw=Morlet(π), β=2, σ=abs)
sf = St(f)
```

The results `sf` are stored in the `ScatteredOut` type; for a two layer scattering transform, it has three output matrices (zeroth, first and second layers).

### Zeroth Layer

The zeroth layer is simply a moving average of the original signal:

```@example ex
plot(sf[0][:, 1, 1], title="Zeroth Layer", legend=false)
```

### First Layer

The first layer is the average of the absolute value of the scalogram:

```@example ex
f1, f2, f3 = getMeanFreq(St) # the mean frequencies for the wavelets in each layer
heatmap(1:size(sf[1], 1), f1[1:end-1], sf[1][:, :, 1]', xlabel="time index", ylabel="Frequency (Hz)", color=:viridis, title="First Layer")
```

### Second Layer

The second layer is where the scattering transform begins to get more involved, and reflects both the frequency of [the envelope](https://en.wikipedia.org/wiki/Analytic_signal#Instantaneous_amplitude_and_phase) surrounding the signal and the frequency of the signal itself.
Visualizing this is also somewhat difficult, since each path in the second layer is indexed by a pair `(s2,s1)`, where `s2` is the frequency used for the second layer wavelet, and `s1` is the frequency used for the first layer wavelet.
To this end, lets make two gifs, the first with the _first_ layer frequency varying with time:

```@example ex
anim = Animation()
for jj = 1:length(f1)-1
    toPlot = dropdims(sf[pathLocs(2, (:, jj))], dims=3)
    heatmap(1:size(sf[2], 1), f2[1:end-1], toPlot', title="index=$(jj), first layer frequency=$(round(f1[jj],sigdigits=4))Hz", xlabel="time", ylabel="Frequency (Hz)", c=cgrad(:viridis, scale=:exp))
    frame(anim)
end
gif(anim, "sliceByFirst.gif", fps=1)
```

By fixing the first layer frequency, we get the scalogram of a single line from the scalogram above.
As the first layer frequency increases, the energy concentrates to the beginning of the signal and increased frequency, and generally decreases.

The second has the _second_ layer frequency varying with time:

```@example ex
anim = Animation()
for jj = 1:length(f2)-1
    toPlot = dropdims(sf[pathLocs(2, (jj, :))], dims=3)
    heatmap(1:size(sf[2], 1), f1[1:end-1], toPlot', title="index=$(jj), second layer frequency=$(round(f2[jj],sigdigits=4))Hz", c=cgrad(:viridis, scale=:exp))
    frame(anim)
end
gif(anim, "sliceBySecond.gif", fps=1)
```

For any fixed second layer frequency, we get approximately the curve in the first layer scalogram, with different portions emphasized, and the overall mass decreasing as the frequency increases, corresponding to the decreasing amplitude of the envelope for the doppler signal.
These plots can also be created using

From the companion package ScatteringPlots.jl, we have the denser representation:

```@example ex
using ScatteringPlots
# Fix: Explicitly use the function from ScatteringPlots to avoid ambiguity
ScatteringPlots.plotSecondLayer(sf, St)
savefig("second.png") #hide
```

![](second.png)

where the frequencies are along the axes, the heatmap gives the largest value across time for that path, and at each path is a small plot of the averaged timecourse.
