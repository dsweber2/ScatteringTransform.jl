""""
    ∇st(st, at,maxDepth=-1)
Take a gradient w.r.t. every coordinate, and store in a scattered array. The shape is (space variable, pathIndices..., outputLocation). For example, if the input signal has size 128
"""
function ∇st(st, at, maxDepth=-1)
    y, back = pullback(st, at)
    ∇ = scattered(map(x->zeros(size(at)[1:ndims(y)]..., 
                               x[ndims(y):end-1]...), size(y)))
    if maxDepth<0
        maxDepth = length(∇)
    end
    for m = 0:maxDepth
        axes(y[2])[1:end-1]
        for is in Iterators.product(axes(y[m])[1:end-1]...)
            grad = scattered(map(x->zeros(size(x)...), y.output))
            p = pathLocs(m, is)
            grad[p] = 1
            ∇[m][:, is...] = back(grad)[1][:,1,1]
        end
    end
    return ∇
end


"""
    plotFirstLayer1D(j,origLoc,cline=:darkrainbow)
"""
function plotFirstLayer1D(j,origLoc,original,cline=:darkrainbow)
    
    space = plot(origLoc[1][:,:,j], line_z=(1:size(origLoc[1],2))',
                 legend=false, colorbar=true, color=cline, 
                 title="first layer gradient wavelet $j varying location")
    org = plot([original original], line_z=([-20;1:size(origLoc[1],2)...])', legend=false, colorbar=true, color=cline) #todo: including the colorbar here is a real hack to get them to line up
    ∇h = heatmap(origLoc[1][:,1:end,j]', xlabel="space",
                 ylabel="wavelet location", title="First layer gradient j=$j")
    ∇̂h = heatmap(log.(abs.(rfft(origLoc[1][:,1:end,j],1)).^2)',xlabel="frequency", 
                     ylabel="wavelet location", title="Log Power Frequency domain j=$j")
    l = @layout [a; b{.1h}; [b c]]
    plot(space, org, ∇h, ∇̂h,layout=l)
end

"""
    gifFirstLayer(origLoc, saveTo="tmp.gif", fps = 2)
"""
function gifFirstLayer(origLoc, firstSig, saveTo="gradientFigures/tmp.gif", fps = 2)
    anim = Animation()
    for j=1:size(origLoc[1])[end]
        plotFirstLayer1D(j, origLoc, firstSig)
        frame(anim)
    end
    gif(anim, saveTo, fps=2)
end




meanWave(wave) = sum(real.(range(0,stop=1,length=size(wave,1)) .* wave),dims=1) ./ sum(real.(wave), dims=1)

function plotSecondLayer1D(loc, origLoc, wave1, wave2, original=false, subsamSz=(128,85,), c=:thermal, lastDiagFreq=true)
    waveUsed = real.(ifftshift(irfft(wave1[:,loc[2]], subsamSz[1]*2)))
    l1wave = plot(waveUsed, legend=false,titlefontsize=8,title="layer 1 ($(loc[2]))")
    annotate!(size(waveUsed,1)*5/6, maximum(waveUsed), Plots.text("freq = $(meanWave(wave1)[loc[2]])"[1:13], 5))
    waveUsed = real.(ifftshift(irfft(wave2[:,loc[1]], subsamSz[2]*2)))
    l2wave = plot(waveUsed,legend=false,titlefontsize=8,title="layer 2 ($(loc[1]))")
    annotate!(size(waveUsed, 1)*5/6, maximum(waveUsed), 
              Plots.text("freq = $(meanWave(wave2)[loc[1]])"[1:13], 5))
    if original!=false
        org = plot([original original], line_z=(1:size(origLoc[1],2))', legend=false, colorbar=true, color=:darkrainbow) #todo: including the colorbar here is a real hack to get them to line up
    end
    ∇heat = heatmap(origLoc[2][:, :, loc...]', ylabel="wavelet location",
                    xlabel="space",color=c, title="second layer gradient ")
    if lastDiagFreq
        ∇plt = heatmap(log.(abs.(rfft(origLoc[2][:,:,loc...],1))), xlabel="wavelet location", ylabel="frequency", color=c, title="varying location, path $(loc[2:-1:1])")
    else
        ∇plt = plot(origLoc[2][:,:,loc...], line_z=(28:-1:1)', legend=false,color=:cividis, title="varying location, path $(loc[2:-1:1])")
    end

    if original!=false
        l = @layout [[a; b{.1h}; [c d]] e]
        return plot(∇heat, org, l1wave, l2wave, ∇plt, layout=l)
    else
        l = @layout [[a; [b c]] d]
        return plot(∇heat, l1wave, l2wave, ∇plt, layout=l)
    end
end

"""
    plotSecondLayer(stw; title="Second Layer results",xVals=-1,yVals=-1,logPower=false, toHeat=nothing, c=cgrad([:blue, :orange], scale=:log), threshold=0)
TODO fix the similarity of these names.
In the case that arbitrary space has been introduced, if you have a title, use `xVals = (.037, .852), yVals = (.056, .939)`, or if you have no title, use `xVals = (.0105, .882), yVals = (.056, .939)`
"""
function plotSecondLayer(stw; title="Second Layer results",xVals=-1,yVals=-1,logPower=false, toHeat=nothing, c=cgrad([:blue, :orange], scale=:log), threshold=0)
    n,m = size(stw[2])[2:3]
    gr(size=2.5 .*(280,180))
    if xVals == -1  &&  title == ""
        xVals = (.0105, .882)
    elseif xVals ==-1
        xVals = (.0005, .889)
    end
    Δx = xVals[2]-xVals[1]
    xrange = range(xVals[1]+Δx/m-Δx/(m+3), 
                   stop = xVals[2]-2*Δx/(m+3)+Δx/m, length=m)
    if yVals == -1
        yVals=(0.0, .995)
    end
    Δy = yVals[2]-yVals[1]
    yrange = range(yVals[1]+Δy/n-Δy/(n+3), stop = yVals[2]-2*Δy/(n+3)+Δy/n, length=n)
    if toHeat == nothing
        toHeat = [norm(stw[2][:,i,j,1]) for i=1:n,j=1:m]
    end
    if logPower
        toHeat = log.(toHeat)
    end
    if title==""
        plt = heatmap(toHeat, yticks=1:n, xticks=1:m, tick_direction=:out,
                      xlabel="Layer 1 index", ylabel="Layer 2 index",c=c)
    else
        plt = heatmap(toHeat, yticks=1:n, xticks=1:m, tick_direction=:out,
                      title=title, xlabel="Layer 1 index", ylabel="Layer 2 index", c=c)
    end
    nPlot = 2
    totalRange = maximum(toHeat)-minimum(toHeat)
    bottom = minimum(toHeat)
    for i=1:n, j=1:m
        if maximum(abs.(stw[2][:,i,j,:]))>threshold
            plt = plot!(stw[2][:,i,j,:], legend=false, subplot=nPlot,
                        bg_inside = cgrad(c)[(toHeat[i,j]-bottom)/totalRange],
                        ticks=nothing, palette=:greys, frame=:box,
                        inset=(1,bbox(xrange[j], yrange[i], Δx/(m+10),
                                      Δy/(n+10),:bottom,:left)))
            nPlot+=1
        end
    end
    plt
end


"""
    addNextPath(addTo,addFrom)
    addNextPath(addFrom)

adds the next nonzero path in addFrom not present in addTo, ordered by layer, and 
then the standard order for julia (dimension 1, then 2, 3,etc). If only handed `addFrom`
it makes a new pathLoc only containing the first non-empty path.

presently it only works for pathLocs whose indices are Boolean Arrays
"""
function addNextPath(addTo::pathLocs{m},addFrom) where m
    shouldWeDo = foldl(checkShouldAdd, zip(addTo.indices, addFrom.indices), init=Tuple{}())
    inds = map(collatingTransform.makeNext, addTo.indices, addFrom.indices, shouldWeDo)
    return pathLocs{m}(inds)
end
function checkShouldAdd(didPrevs,current)
    if !any(didPrevs) && current[1] != nothing
        addLoc = findfirst(current[1].!=current[2])
        if addLoc!=nothing
            return (didPrevs..., true)
        else
            return (didPrevs..., false)
        end
    else
        return (didPrevs..., false)
    end
end
function makeNext(current, fillWith, shouldDo)
    if shouldDo
        addLoc = findfirst(fillWith.!=current)
        newVersion = copy(current)
        newVersion[addLoc] = true
        return newVersion
    else
        return current
    end
end

addNextPath(addFrom::pathLocs{m}) where m = pathLocs{m}(foldl(makeSingle, addFrom.indices, init=Tuple{}()))

makeSingle(prev,x::Nothing) = (prev...,nothing)
makeSingle(x::Nothing) = (nothing,)
function makeSingle(x::BitArray)
    almostNull = falses(size(x))
    almostNull[findfirst(x)] = true
    return (prev..., almostNull)
end
function makeSingle(prev,x::BitArray)
    almostNull = falses(size(x))
    # only set this layer to true if all the previous are nothing
    if typeof(prev) <: Tuple{Vararg{Nothing}} #|| !any.(any.(prev[prev .!=nothing]))
        almostNull[findfirst(x)] = true
    end
    return (prev..., almostNull)
end

