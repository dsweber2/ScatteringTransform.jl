using ScatteringTransform
function sizes(func::Function, rate::Array{T}, sizeX::Int64) where T<:Real
  if func == bspline
    subsamp = zeros(Int,length(rate)+1); subsamp[1] = sizeX[1]
    for (i,rat) in enumerate(rate)
      subsamp[i+1] = floor(Int, subsamp[i]./rat)
    end
  elseif func == bilinear
    subsamp = [Int64( foldl((x,y)->ceil(x/y), sizeX, rate[1:i])) for i=0:length(rate)-1]
  else
    error("Sorry, that hasn't been implemented yet")
  end
  subsamp
end
function numScales(c::CFWA, n::S) where S<:Integer

    J1=floor(Int64,(log2(n)-2)*c.scalingFactor)
    #....construct time series to analyze, pad if necessary
    if eltypes(c) == WT.padded
        base2 = round(Int,log(n)/log(2));   # power of 2 nearest to N
        n = n+2^(base2+1)-n
    elseif eltypes(c) == WT.DEFAULT_BOUNDARY
        n= 2*n
    else
        n=n
    end

    return J1+2-c.averagingLength
end
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
