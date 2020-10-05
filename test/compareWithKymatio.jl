using Revise, ScatteringTransform, Plots, LinearAlgebra
pyplot()

k = 2^9
t = (-1+1/k):1/k:1; size(t) # interval
n = 20       # degree of polynomial
testSignal = hcat([t.^i for i=0:n]...)* randn(n+1)
plot(t, testSignal)


layers = stParallel(2, length(testSignal))
# 11 is *way* too far out
layers = stParallel(2, length(testSignal), CWTType = WT.Morlet(11), nScales=[8 for i=1:3])
layers.shears[1].averagingLength

# let's look at the actual filters that are used
n = ScatteringTransform.getNs(size(testSignal), layers)
i=1

layers.shears[i]
daughters,ξ = computeWavelets(n[i], layers.shears[i])
layers.shears[1].frameBound
ω = [0:ceil(Int, n[1]); -floor(Int,n[1])+1:-1]*2π;
plot(daughters[:, :], legend=false)
plot(daughters[:, end], legend=false)

plt = Array{Any,1}(undef,4)
for (k,x) in enumerate([(:none,"Unnormalized (may be using this one)"), (:sqrtScaling,
                                                 "2-norm Scaling"),
                        (:absScaling, "1-norm scaling"),
                        (:maxOne, "∞-norm scaling (they use)")])
    tmpDaughters = zeros(33,n[1]+1)
    for i=0:32
        tmpDaughters[i+1, :] = ScatteringTransform.Daughter(layers.shears[1],
                                                            2.0^(2+i/8), ω,
                                                            x[1])[1:(n[1]+1)]
    end
    plt[k] = plot(tmpDaughters[:,:]', legend=false, title="$(x[2])")
end
plot(plt...,layout=(4,1))
savefig("possibleScalings.pdf")
norm(ScatteringTransform.Daughter(layers.shears[i], 2.0^(3/8), ω)[1:(n[1]+1)])
layers.shears[i]

stResult = st(testSignal, layers, absType())





