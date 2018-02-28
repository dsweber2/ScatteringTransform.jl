# tests for the various forms of layeredTransform for the ShatteringTransform
using ScatteringTransform
using Shearlab, Interpolations, Wavelets, JLD, MAT, Plots, LaTeXStrings
# scattered2D tests
X=randn(1024)
lT = layeredTransform(2,randn(1024,1024))
Shearlab.SLsheardec2D(X,lT.shears[1])
scattered2D(2,[1024,512,128],[1024,512,128], [2,17,17],Complex128)

scattered2D(lT)
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

# # tests for the various forms of layeredTransform for the 1D ContinuousWaveletClass transforms
# f = randn(10214)
# m=3
# layeredTransform(m, f, subsampling=[8,4,2,1], nScales=[16,8,8,8], CWTType=WT.dog2, averagingLength=[16,4,4,2],averagingType=[:Mother for i=1:(m+1)],boundary=[WT.DEFAULT_BOUNDARY for i=1:(m+1)])
# layeredTransform(3, length(f), [2, 2, 2, 2], [2, 2, 2, 2], WT.dog2)
# layeredTransform(3, length(f), 8, [2, 2, 2, 2], WT.dog2)
# layeredTransform(3, length(f), [2, 2, 2, 2], 2, WT.dog2)
# layeredTransform(3, length(f), 8, 2, WT.dog2)
#
# layers = layeredTransform(3, length(f), 8, 4)
# # plot(abs.(ψ'))
# cwt(f,layers.shears[1])
#
# results = scattered1D(layers,f)
# [(1:3)';(4:6)';(7:9)']
# reshape([(1:3)';(4:6)';(7:9)'],(9))
# layeredTransform(3, f, [2, 2, 2, 2], [2, 2, 2, 2], WT.dog2)
# layeredTransform(3, f, 8, [2, 2, 2, 2], WT.dog2)
# layeredTransform(3, f, [2, 2, 2, 2], 2, WT.dog2)
# layeredTransform(3, f, 8, 2, WT.dog2)
#
# layeredTransform(3, f, [2, 2, 2, 2], WT.dog2)
# layeredTransform(3, f, 7, WT.dog2)
# layeredTransform(3, length(f), [7, 7, 7, 7], WT.dog2)
# layeredTransform(3, length(f), 7, WT.dog2)
#
# layeredTransform(3,f,WT.dog2)
#
# layeredTransform(3, length(f), [2, 2, 2, 2], [2, 2, 2, 2])
# layeredTransform(3, length(f), 8, [2, 2, 2, 2])
# layeredTransform(3, length(f), [2, 2, 2, 2], 2)
# layeredTransform(3, length(f), 8, 2)
#
# layeredTransform(3, f, [2, 2, 2, 2], [2, 2, 2, 2])
# layeredTransform(3, f, 8, [2, 2, 2, 2])
# layeredTransform(3, f, [2, 2, 2, 2], 2)
# layeredTransform(3, f, 8, 2)
#
# layeredTransform(3, f, [2, 2, 2, 2])
# layeredTransform(3, f, 7)
# layeredTransform(3, length(f), [7, 7, 7, 7])
# layeredTransform(3, length(f), 7)
#
# layeredTransform(3,f)
#
#
# # scratch
#
# X = testfunction(1532, "HeaviSine")
# layers = layeredTransform(3, X, 8, [8,4,2,1])
# nonlinear = abs
# subsam = bspline
#
# results = scattered1D(layers, X)
# # for (i,layer) in enumerate(layers.shears)
#   # println("layer $i $layer")
# # end
# i=1
# cur = results.data[i]
# λ=1
# output=cwt(cur[:,λ], layers.shears[i])'
# innput = output[:,1]
# using ScatteringTransform
# using Shearlab, Interpolations, Wavelets, JLD, MAT, Plots, LaTeXStrings
# bspline(innput,8)
# itp = interpolate(innput, BSpline(Quadratic(Reflect())), OnGrid())
# itp[linspace(1,length(innput),floor(length(innput)./8))]
#
# bspline(output[:,1], layers.subsampling[i])
# bspline(X, layers.subsampling[i])
# j=1
# results.data[i+1][:,(λ-1)*(size(output,2)-1)+j] = nonlinear.(subsam(output[:,j], layers.subsampling[i]))
# for (i,layer) in enumerate(layers.shears)
#   cur = results.data[i] #data from the previous layer
#   if i<=layers.m
#     # println("i=$i")
#     for λ = 1:size(cur,2)
#       # first perform the continuous wavelet transform on the data from the previous layer
#       output = cwt(cur[:,λ], layers.shears[i])'
#       # subsample each example, then apply the non-linearity
#       for j = 1:size(output,2)-1
#         if subsam== bspline
#           # println("accessing results.data at i+1=$(i+1), [:,$((λ-1)*(size(output,2)-1)+j)], j=$j")
#           results.data[i+1][:,(λ-1)*(size(output,2)-1)+j] = nonlinear.(subsam(output[:,j], layers.subsampling[i]))
#         else
#           error("the function $subsam isn't defined as a subsampling method at the moment")
#         end
#       end
#       # println("accessing results.output at $i, [:,$λ]")
#       results.output[i][:,λ] = Array{Float64,1}(real(subsam(output[:,end], layers.subsampling[i])))
#     end
#   else
#     # TODO: This is not an efficient implementation to get the last layer of output. There are several places where m+1 is substituted for m in the definition of shears to accomodate it. Data is shrunk by a layer in the sheared array
#     println("i too big i=$(i)")
#     for λ = 1:size(cur,3)
#       output = cwt(cur[:,λ], layers.shears[i])'
#       results.output[i][:,λ] = Array{Float64,1}(real(subsam(output[:,end], layers.subsampling[i])))
#     end
#   end
# end
#
#
#
# results
# @time output = st(X, layers)
# output2 = st(randn(1532), layers)
# heatmap(max.(-10,log.( abs.(output.data[2]))))
# plot(abs.(output.output[1]))
# heatmap(max.(-100,log.(abs.(output.output[2]))))
# heatmap(max.(-100,log.(abs.(output.output[3]))))
# heatmap(max.(-100,log.(abs.(output.output[4]))))
# heatmap(max.(-100,log.(abs.(output2.output[4]))))
# heatmap(abs.(output.data[2]))
#
# x0 = testfunction(1532, "HeaviSine")
# t= -10:.01:10
# x1 = t.^2
# plot(x0)
# c=  wavelet(WT.morl,8,10)
# cwtx=cwt(x0,c)
# cwtAVEx= cwt(x0,c)
# daughters, ω =getScales(1532,c,4)
# heatmap(log.(abs.(cwtx)),xlabel="time",ylabel="frequency")
# heatmap(log.(abs.(cwt(x1,c))),xlabel="time",ylabel="frequency")
# plot(real.(cwtx[1,:]))
# plot(abs.(cwtAVEx[1,:]))
# plot(abs.(fft(x0)))
# layers = layeredTransform(3, length(f), [2, 2, 2, 2], [2, 2, 2, 2],WT.dog1)
# Y= f;; c= layers.shears[1]; averagingLength=11; averagingType=:Mother
#
# # create an ω that, when fed to the daughter function, translates the scaled function to the origin
#
# c
# c.σ[1]
#
# Y= f;; c= layers.shears[1]; averagingLength=11; averagingType=:Mother
# J1=NaN
#
#
# plot(abs.(daughters)')
# plot!(sum(abs.(daughters)'.^2,2))
# daughters[1,:]
# size(daughters)
# # TODO: should the daughters form a partition of unity on the positive axis?
#
#
# layers = layeredTransform(3, length(f), [2, 2, 2, 2], [2, 2, 2, 2], WT.morl)
# Y= f;; c= layers.shears[1]; averagingLength=2; averagingType=:Dirac
#
# plot(findAveraging(c, averagingLength,averagingType,ω)[1:20])
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# ShatteringTransform()
# # mucking around with Wavelets.jl
# n=2053
# plot(x)
# awef = cwt(x, wavelet(WT.morl))
# heatmap(abs.(awef))
# x = testfunction(n,"Doppler")
# wt = wavelet(WT.db2)
# y = dwt(x, wt, maxtransformlevels(x))
# y2 = dwt(x, WT.scale(wt,1/2))
# y4 = dwt(x, WT.scale(wt,1/4))
# plot(y, label="wt")
# plot(y2, label="wt2")
# plot(y4, label="wt4")
# d,l = wplotdots(y, 0.1, n)
# Ap = wplotim(y)
# heatmap(A)
# heatmap(Ap)
# plot(y,label="discrete")
# plot(x)
#
#
# N = 2^10
#
#
# h = WT.qmf(wavelet(WT.db2))
# plot(h)
#
# # Find the initial points
# H = Array{eltype(h)}(length(h),length(h))
# for i=1:length(h)
#     for j=1:length(h)
#         if 2(i-1)-(j-1) >= 0 && 2i-j <= length(h)
#             H[i,j] = h[2i-j]
#         else
#             H[i,j] = 0
#         end
#     end
# end
# eigval, evec = eig(H)
# ϕ = evec[:,eigval.==1/sqrt(2)]
# ϕ = ϕ./sum(ϕ)
# plot(ϕ)
# plot(0:3,h)
# plot!(0:4/64:3, resample(resample(resample(resample(h,2.025),2.025),2.025),2.025))
# FIRFilter(h,2//1)
# resample(h, 2//1, resample_filter(2//1))
# FIRRational(h,2//1)
# resample(resample(h,2//1),2//1)
# plot(resample(resample(h,2.0),2.0))
# plot(filt(FIRFilter(2//1), h))
# scatter!(1:2:8,h)
# # keep going until we get to over N points
# ϕp = zeros(eltype(h), 2*length(ϕ))
# # upsample h
# hp = upsample(h)
# # smooth out the zeros we padded h with
# hp =
# plot(sinc(-π:.05:π))
# plot(hp⋆ϕ/sum(hp⋆ϕ))
# h
# for m=1:length(ϕp)
#     # these are from the previous layer
#     if m%2==1
#         ϕp[m] = ϕ[div(m+1,2)]
#     else
#         for n=1:length(ϕ)
#             if 2(n-1)-(m-1) >= 0 && 2n-m <= length(h)
#                 ϕp[m] += h[2n-m].*ϕ[div(n,2)]
#             end
#         end
#     end
# end
# ϕp = sum(ϕp)*ϕp
# plot(ϕp)
# ϕp
# # resample back down to N points
# eval==sqrt(2)
# plot(evec[:,2])
# ϕ =
# 1/sqrt(sqrt(2))
# svd(H)
#
#
#
#
# sqrt(2)*h
