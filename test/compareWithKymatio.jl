using ScatteringTransform, Plots, LinearAlgebra
pyplot()

k = 256
t = (-1+1/k):1/k:1; size(t) # interval
n = 20       # degree of polynomial
testSignal = hcat([t.^i for i=0:n]...)* randn(n+1)
plot(t, testSignal)


layers = layeredTransform(2, length(testSignal))


# let's look at the actual filters that are used
n = ScatteringTransform.getNs(size(testSignal), layers)
i=1
daughters = computeWavelets(n[i], layers.shears[i])
layers.shears[1].frameBound
plot(daughters,legend=false)


stResult = st(testSignal, layers, absType())






