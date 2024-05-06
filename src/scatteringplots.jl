"""
    plotFirstLayer1D(j,origLoc,cline=:darkrainbow)
"""
function plotFirstLayer1D(j, origLoc, original, cline=:darkrainbow)

    space = plot(origLoc[1][:, :, j], line_z=(1:size(origLoc[1], 2))',
        legend=false, colorbar=true, color=cline,
        title="first layer gradient wavelet $j varying location")
    org = plot([original original], line_z=([-20; 1:size(origLoc[1], 2)...])', legend=false, colorbar=true, color=cline) # todo: including the colorbar here is a real hack to get them to line up
    ∇h = heatmap(origLoc[1][:, 1:end, j]', xlabel="space",
        ylabel="wavelet location", title="First layer gradient j=$j")
    ∇̂h = heatmap(log.(abs.(rfft(origLoc[1][:, 1:end, j], 1)) .^ 2)', xlabel="frequency",
        ylabel="wavelet location", title="Log Power Frequency domain j=$j")
    l = Plots.@layout [a; b{0.1h}; [b c]]
    plot(space, org, ∇h, ∇̂h, layout=l)
end

"""
    gifFirstLayer(origLoc, saveTo="tmp.gif", fps = 2)
"""
function gifFirstLayer(origLoc, firstSig, saveTo="gradientFigures/tmp.gif", fps=2)
    anim = Animation()
    for j = 1:size(origLoc[1])[end]
        plotFirstLayer1D(j, origLoc, firstSig)
        frame(anim)
    end
    gif(anim, saveTo, fps=2)
end

meanWave(wave) = sum(real.(range(0, stop=1, length=size(wave, 1)) .* wave), dims=1) ./ sum(real.(wave), dims=1)

function plotSecondLayer1D(loc, origLoc, wave1, wave2, original=false, subsamSz=(128, 85,), c=:thermal, lastDiagFreq=true)
    waveUsed = real.(ifftshift(irfft(wave1[:, loc[2]], subsamSz[1] * 2)))
    l1wave = plot(waveUsed, legend=false, titlefontsize=8, title="layer 1 ($(loc[2]))")
    annotate!(size(waveUsed, 1) * 5 / 6, maximum(waveUsed), Plots.text("freq = $(meanWave(wave1)[loc[2]])"[1:13], 5))
    waveUsed = real.(ifftshift(irfft(wave2[:, loc[1]], subsamSz[2] * 2)))
    l2wave = plot(waveUsed, legend=false, titlefontsize=8, title="layer 2 ($(loc[1]))")
    annotate!(size(waveUsed, 1) * 5 / 6, maximum(waveUsed),
        Plots.text("freq = $(meanWave(wave2)[loc[1]])"[1:13], 5))
    if original != false
        org = plot([original original], line_z=(1:size(origLoc[1], 2))', legend=false, colorbar=true, color=:darkrainbow) # todo: including the colorbar here is a real hack to get them to line up
    end
    ∇heat = heatmap(origLoc[2][:, :, loc...]', ylabel="wavelet location",
        xlabel="space", color=c, title="second layer gradient ")
    if lastDiagFreq
        ∇plt = heatmap(log.(abs.(rfft(origLoc[2][:, :, loc...], 1))), xlabel="wavelet location", ylabel="frequency", color=c, title="varying location, path $(loc[2:-1:1])")
    else
        ∇plt = plot(origLoc[2][:, :, loc...], line_z=(28:-1:1)', legend=false, color=:cividis, title="varying location, path $(loc[2:-1:1])")
    end

    if original != false
        l = Plots.@layout [[a; b{0.1h}; [c d]] e]
        return plot(∇heat, org, l1wave, l2wave, ∇plt, layout=l)
    else
        l = Plots.@layout [[a; [b c]] d]
        return plot(∇heat, l1wave, l2wave, ∇plt, layout=l)
    end
end

"""
    plotSecondLayer(stw; title="Second Layer results", xVals=-1, yVals=-1, logPower=true, toHeat=nothing, c=cgrad(:viridis, [0,.9]), threshold=0, linePalette=:greys, minLog=NaN, kwargs...)
TODO fix the similarity of these names.
xVals and yVals give the spacing of the grid, as it doesn't seem to be done
correctly by default. xVals gives the distance from the left and the right
as a tuple, while yVals gives the distance from the top and the bottom,
also as a tuple. Default values are `xVals = (.037, .852), yVals = (.056, .939)`, or if you have no title, use `xVals = (.0105, .882), yVals = (.056, .939)`
If you have no colorbar, set `xVals = (.0015, .997), yVals = (.002, .992)`
In the case that arbitrary space has been introduced, if you have a title, use `xVals = (.037, .852), yVals = (.056, .939)`, or if you have no title, use `xVals = (.0105, .882), yVals = (.056, .939)`
"""
function plotSecondLayer(stw::ScatteredOut, st; kwargs...)
    secondLayerRes = stw[2]
    if ndims(secondLayerRes) > 3
        return plotSecondLayer(secondLayerRes[:, :, :, 1], st; kwargs...)
    else
        return plotSecondLayer(secondLayerRes, st; kwargs...)
    end
end

function plotSecondLayer(stw, st; title="Second Layer results", xVals=-1, yVals=-1, logPower=true, toHeat=nothing, c=cgrad(:viridis, [0, 0.9]), threshold=0, freqsigdigits=3, linePalette=:greys, minLog=NaN, subClims=(Inf, -Inf), δt=1000, firstFreqSpacing=nothing, secondFreqSpacing=nothing, transp=true, labelRot=30, xlabel=nothing, ylabel=nothing, frameTypes=:box, miniFillAlpha=0.5, kwargs...)
    n, m = size(stw)[2:3]
    freqs = getMeanFreq(st, δt)
    freqs = map(x -> round.(x, sigdigits=freqsigdigits), freqs)[1:2]
    gr(size=2.5 .* (280, 180))
    if !(typeof(c) <: PlotUtils.ContinuousColorGradient)
        c = cgrad(c)
    end
    if toHeat == nothing
        toHeat = [norm(stw[:, i, j, 1]) for i = 1:n, j = 1:m]
    end
    if firstFreqSpacing == nothing
        firstFreqSpacing = 1:m
    end
    if secondFreqSpacing == nothing
        secondFreqSpacing = 1:n
    end
    # transp means transpose so that the second layer frequency is along the x-axis
    if transp
        xTicksFreq = (secondFreqSpacing, freqs[2][secondFreqSpacing])
        yTicksFreq = (firstFreqSpacing, freqs[1][firstFreqSpacing])
        freqs = reverse(freqs)
        nTmp = n
        n = m
        m = nTmp
        toHeat = toHeat'
        xInd = 2
        yInd = 1
        stw = permutedims(stw, (1, 3, 2, (4:ndims(stw))...))
    else
        xTicksFreq = (firstFreqSpacing, freqs[1][firstFreqSpacing])
        yTicksFreq = (secondFreqSpacing, freqs[2][secondFreqSpacing])
        (1:m, freqs[1])
        xInd = 1
        yInd = 2
    end
    if xVals == -1 && title == ""
        xVals = (0.0105, 0.882)
    elseif xVals == -1
        xVals = (0.002, 0.880)
    end
    Δx = xVals[2] - xVals[1]
    xrange = range(xVals[1] + Δx / m - Δx / (m + 3),
        stop=xVals[2] - 2 * Δx / (m + 3) + Δx / m, length=m)
    if yVals == -1
        yVals = (0.0, 0.995)
    end
    Δy = yVals[2] - yVals[1]
    yrange = range(yVals[1] + Δy / n - Δy / (n + 3), stop=yVals[2] - 2 * Δy / (n + 3) + Δy / n, length=n)
    if logPower
        toHeat = log10.(toHeat)
        if !isnan(minLog)
            toHeat = max.(minLog, toHeat)
        end
    end
    bottom = min(minimum(toHeat), subClims[1])
    top = max(subClims[2], maximum(toHeat))
    totalRange = top - bottom
    # substitute given x and y labels if needed
    if isnothing(xlabel)
        xlabel = "Layer $(xInd) frequency (Hz)"
    end
    if isnothing(ylabel)
        ylabel = "Layer $(yInd) frequency (Hz)"
    end

    if title == ""
        plt = heatmap(toHeat; yticks=yTicksFreq, xticks=xTicksFreq, tick_direction=:out, rotation=labelRot,
            xlabel=xlabel, ylabel=ylabel, c=c, clims=(bottom, top), kwargs...)
    else
        plt = heatmap(toHeat; yticks=yTicksFreq, xticks=xTicksFreq, tick_direction=:out, rotation=labelRot,
            title=title, xlabel=xlabel, ylabel=ylabel, c=c, clims=(bottom, top), kwargs...)
    end
    nPlot = 2
    for i in 1:n, j in 1:m
        if maximum(abs.(stw[:, i, j, :])) > threshold
            plt = plot!(stw[:, i, j, :], legend=false, subplot=nPlot,
                bg_inside=c[(toHeat[i, j]-bottom)/totalRange],
                ticks=nothing, palette=linePalette, frame=frameTypes,
                inset=(1, bbox(xrange[j], yrange[i], Δx / (m + 10),
                    Δy / (n + 10), :bottom, :left)), fill=0, fillalpha=miniFillAlpha)
            nPlot += 1
        end
    end
    plt
end

function jointPlot(thingToPlot, thingName, cSymbol, St; sharedColorScaling=:exp, targetExample=1, δt=1000, freqigdigits=3, sharedColorbar=true, extraPlot=nothing, allPositive=false, logPower=false, kwargs...)
    if sharedColorbar
        clims = (min(minimum.(thingToPlot)...), max(maximum.(thingToPlot)...))
        climszero = clims
        climsfirst = clims
        climssecond = clims
        toHeat = [norm(thingToPlot[2][:, i, j, 1], Inf) for i = 1:size(thingToPlot[2], 2), j = 1:size(thingToPlot[2], 3)]
    else
        climszero = (min(minimum.(thingToPlot[0])...), max(maximum.(thingToPlot[0])...))
        climsfirst = (min(minimum.(thingToPlot[1])...), max(maximum.(thingToPlot[1])...))
        climssecond = (min(minimum.(thingToPlot[2])...), max(maximum.(thingToPlot[2])...))
        toHeat = [norm(thingToPlot[2][:, i, j, 1], Inf) for i = 1:size(thingToPlot[2], 2), j = 1:size(thingToPlot[2], 3)]
    end
    firstLay = thingToPlot[1][:, :, targetExample]'
    zeroLay = thingToPlot[0][:, :, targetExample]'
    toHeat[toHeat.==0] .= -Inf    # we would like zeroes to not actually render
    firstLay[firstLay.==0] .= -Inf # for either layer

    # adjust the other parts to be log if logPower is true
    if logPower && allPositive
        absThing = map(x -> abs.(x), (thingToPlot[0], thingToPlot[1], thingToPlot[2]))
        clims = (min(minimum.(absThing)...), max(maximum.(absThing)...))
        firstLay = log10.(firstLay)
        zeroLay = log10.(abs.(zeroLay))
        climszero = log10.(clims)
        climsfirst = log10.(clims)
        climssecond = log10.(clims)
    elseif logPower
        error("not currently plotting log power and negative values")
    end

    if allPositive
        c = cgrad(cSymbol, scale=sharedColorScaling)
        cSecond = cgrad(cSymbol)
    else
        zeroAt = -climssecond[1] / (climssecond[2] - climssecond[1]) # set the mid color switch to zero
        c = cgrad(cSymbol, [0, zeroAt], scale=sharedColorScaling)
        cSecond = cgrad(cSymbol, [0, zeroAt])
    end

    # define the spatial locations as they correspond to the input
    spaceLocs = range(1, size(St)[1], length=length(zeroLay))

    p2 = plotSecondLayer(thingToPlot[2][:, :, :, targetExample], St; title="Second Layer", toHeat=toHeat, logPower=logPower, c=c, clims=climssecond, subClims=climssecond, cbar=false, xVals=(0.000, 0.993), yVals=(0.0, 0.994), transp=true, xlabel="", kwargs...)
    freqs = getMeanFreq(St, δt)
    freqs = map(x -> round.(x, sigdigits=freqigdigits), freqs)
    p1 = heatmap(firstLay, c=c, title="First Layer", clims=climsfirst, cbar=false, yticks=((1:size(firstLay, 1)), ""), xticks=((1:size(firstLay, 2)), ""), bottom_margin=-10Plots.px)
    p0 = heatmap(spaceLocs, 1:1, zeroLay, c=c, xlabel="time (ms)\nZeroth Layer", clims=climszero, cbar=false, yticks=nothing, top_margin=-10Plots.px, bottom_margin=10Plots.px)
    colorbarOnly = scatter([0, 0], [0, 1], zcolor=[0, 3], clims=climssecond, xlims=(1, 1.1), xshowaxis=false, yshowaxis=false, label="", c=c, grid=false, framestyle=:none)
    if extraPlot == nothing
        # extraPlot = scatter([0,0], [0,1], legend=false, grid=false, x="Layer 2 frequency", foreground_color_subplot=:white, top_margin=-10Plots.px, showaxis=false, yticks=nothing)
        extraPlot = plot(xlabel="Layer 2 frequency (Hz)", grid=false, xticks=(1:5, ""), showaxis=false, yticks=nothing, bottom_margin=-10Plots.px, top_margin=-20Plots.px)
        # extraPlot = heatmap(zeroLay, c=c, xlabel="location\nZeroth Layer", clims=climszero, cbar=false, yticks=nothing, top_margin=-10Plots.px, bottom_margin=10Plots.px)
    end
    titlePlot = plot(title=thingName, grid=false, showaxis=false, xticks=nothing, yticks=nothing, bottom_margin=-10Plots.px)
    lay = Plots.@layout [o{0.00001h}; [[a b; c{0.1h} d{0.1h}] b{0.04w}]]
    plot(titlePlot, p2, p1, extraPlot, p0, colorbarOnly, layout=lay)
end
