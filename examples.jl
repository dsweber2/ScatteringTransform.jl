using ScatteringTransform

[ones(numScales(layers.shears[i],n[i])length(layers.shears)) for i=1:length(layers)]
layers
n = sizes(bspline, layers.subsampling, length(X)) #TODO: if you add another subsampling method in 1D, this needs to change
stType = "full"
q = [numScales(layers.shears[i],n[i]) for i=1:layers.m+1]
s1 = layers.shears[1].scalingFactor
s1 = layers.shears[2].scalingFactor
m=3
counts = zeros(Int64,m+1)
counts[1]=1
for i=2:length(q)
  for a = 1:q[i-1]
    for b = 1:q[i]
      # check to see if the next layer has a larger scale/smaller frequency
      if b./(layers.shears[i].scalingFactor) <= a./(layers.shears[i-1].scalingFactor)
        counts[i]+=1
      end
    end
  end
end
sum([54-i+1 for i=1:30])
54+53+52
counts
layers.shears[1].scalingFactor
layers.shears[1]
