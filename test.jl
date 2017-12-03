# tests for the various forms of layeredTransform for the ShatteringTransform
layeredTransform(3,1000,1000,[3 for i=1:4],[1.0, 1.0, 1.0, 1.0])
X = randn(1000,1000)
layeredTransform(2,X,[2 for i=1:3], [2 for i=1:3])
layeredTransform(2,X,[2 for i=1:3], 2)
layeredTransform(2,X,2,[2 for i=1:3])
layeredTransform(2,X,2,3)
layeredTransform(2,X,[2, 2, 2])
layeredTransform(2,X,2)
layeredTransform(X,2)
layeredTransform(2,X)

# tests for the various forms of layeredTransform for the 1D ContinuousWaveletClass transforms
f = randn(1000)
layeredTransform(3, length(f), [2, 2, 2, 2], [2, 2, 2, 2], WT.dog2)
layeredTransform(3, length(f), 8, [2, 2, 2, 2], WT.dog2)
layeredTransform(3, length(f), [2, 2, 2, 2], 2, WT.dog2)
layeredTransform(3, length(f), 8, 2, WT.dog2)

layeredTransform(3, f, [2, 2, 2, 2], [2, 2, 2, 2], WT.dog2)
layeredTransform(3, f, 8, [2, 2, 2, 2], WT.dog2)
layeredTransform(3, f, [2, 2, 2, 2], 2, WT.dog2)
layeredTransform(3, f, 8, 2, WT.dog2)

layeredTransform(3, f, [2, 2, 2, 2], WT.dog2)
layeredTransform(3, f, 7, WT.dog2)
layeredTransform(3, length(f), [7, 7, 7, 7], WT.dog2)
layeredTransform(3, length(f), 7, WT.dog2)

layeredTransform(3,f,WT.dog2)

layeredTransform(3, length(f), [2, 2, 2, 2], [2, 2, 2, 2])
layeredTransform(3, length(f), 8, [2, 2, 2, 2])
layeredTransform(3, length(f), [2, 2, 2, 2], 2)
layeredTransform(3, length(f), 8, 2)

layeredTransform(3, f, [2, 2, 2, 2], [2, 2, 2, 2])
layeredTransform(3, f, 8, [2, 2, 2, 2])
layeredTransform(3, f, [2, 2, 2, 2], 2)
layeredTransform(3, f, 8, 2)

layeredTransform(3, f, [2, 2, 2, 2])
layeredTransform(3, f, 7)
layeredTransform(3, length(f), [7, 7, 7, 7])
layeredTransform(3, length(f), 7)

layeredTransform(3,f)


# scratch


x0 = testfunction(1532, "HeaviSine")
t= -10:.01:10
x1 = t.^2
plot(x0)
c=  wavelet(WT.morl,8,10)
cwtx=cwt(x0,c)
cwtAVEx= cwt(x0,c)
daughters, ω =getScales(1532,c,4)
heatmap(log.(abs.(cwtx)),xlabel="time",ylabel="frequency")
heatmap(log.(abs.(cwt(x1,c))),xlabel="time",ylabel="frequency")
plot(real.(cwtx[1,:]))
plot(abs.(cwtAVEx[1,:]))
plot(abs.(fft(x0)))
layers = layeredTransform(3, length(f), [2, 2, 2, 2], [2, 2, 2, 2],WT.dog1)
Y= f;; c= layers.shears[1]; averagingLength=11; averagingType=:Mother

# create an ω that, when fed to the daughter function, translates the scaled function to the origin

c
c.σ[1]

Y= f;; c= layers.shears[1]; averagingLength=11; averagingType=:Mother
J1=NaN


plot(abs.(daughters)')
plot!(sum(abs.(daughters)'.^2,2))
daughters[1,:]
size(daughters)
# TODO: should the daughters form a partition of unity on the positive axis?


layers = layeredTransform(3, length(f), [2, 2, 2, 2], [2, 2, 2, 2], WT.morl)
Y= f;; c= layers.shears[1]; averagingLength=2; averagingType=:Dirac

plot(findAveraging(c, averagingLength,averagingType,ω)[1:20])




















ShatteringTransform()
# mucking around with Wavelets.jl
n=2053
plot(x)
awef = cwt(x, wavelet(WT.morl))
heatmap(abs.(awef))
x = testfunction(n,"Doppler")
wt = wavelet(WT.db2)
y = dwt(x, wt, maxtransformlevels(x))
y2 = dwt(x, WT.scale(wt,1/2))
y4 = dwt(x, WT.scale(wt,1/4))
plot(y, label="wt")
plot(y2, label="wt2")
plot(y4, label="wt4")
d,l = wplotdots(y, 0.1, n)
Ap = wplotim(y)
heatmap(A)
heatmap(Ap)
plot(y,label="discrete")
plot(x)


N = 2^10


h = WT.qmf(wavelet(WT.db2))
plot(h)

# Find the initial points
H = Array{eltype(h)}(length(h),length(h))
for i=1:length(h)
    for j=1:length(h)
        if 2(i-1)-(j-1) >= 0 && 2i-j <= length(h)
            H[i,j] = h[2i-j]
        else
            H[i,j] = 0
        end
    end
end
eigval, evec = eig(H)
ϕ = evec[:,eigval.==1/sqrt(2)]
ϕ = ϕ./sum(ϕ)
plot(ϕ)
plot(0:3,h)
plot!(0:4/64:3, resample(resample(resample(resample(h,2.025),2.025),2.025),2.025))
FIRFilter(h,2//1)
resample(h, 2//1, resample_filter(2//1))
FIRRational(h,2//1)
resample(resample(h,2//1),2//1)
plot(resample(resample(h,2.0),2.0))
plot(filt(FIRFilter(2//1), h))
scatter!(1:2:8,h)
# keep going until we get to over N points
ϕp = zeros(eltype(h), 2*length(ϕ))
# upsample h
hp = upsample(h)
# smooth out the zeros we padded h with
hp =
plot(sinc(-π:.05:π))
plot(hp⋆ϕ/sum(hp⋆ϕ))
h
for m=1:length(ϕp)
    # these are from the previous layer
    if m%2==1
        ϕp[m] = ϕ[div(m+1,2)]
    else
        for n=1:length(ϕ)
            if 2(n-1)-(m-1) >= 0 && 2n-m <= length(h)
                ϕp[m] += h[2n-m].*ϕ[div(n,2)]
            end
        end
    end
end
ϕp = sum(ϕp)*ϕp
plot(ϕp)
ϕp
# resample back down to N points
eval==sqrt(2)
plot(evec[:,2])
ϕ =
1/sqrt(sqrt(2))
svd(H)




sqrt(2)*h
