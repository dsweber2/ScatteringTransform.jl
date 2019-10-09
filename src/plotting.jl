########################################################
########################################################
# PLOTTING
########################################################
########################################################

@doc """
    plot = comparePathsChildren(thinResult::Array{<:Real,N}, path::pathType, layers::layeredTransform; outputSubsample=(-1,3), names=["" for i=1:size(thinResult,1)], colorScheme=:curl, scale::Symbol=:log, title="",titleSize=12) where N
    comparePathsChildren(thickResult::scattered{T,N}, path::pathType, layers::layeredTransform; outputSubsample=(-1,3), names=["" for i=1:size(thinResult,1)], colorScheme=:curl, scale::Symbol=:log, title="",titleSize=12)

compares the children of the given path across the first dimensions; don't try to plot too many entries simultaneously. scale gives the color scaling; should be either :log or :linear. path should be a pathType representing the parent.
If pathType is ommitted, it will make a plot for each and save those plots in a directory with the given name
"""
# TODO: only works for depth of 2 at the moment
function comparePathsChildren(thinResult::AbstractArray{<:Real,N}, layers::layeredTransform; outputSubsample=(-1,3), saveDirectory="", names=["" for i=1:size(thinResult,1)], colorScheme=:curl, scale::Symbol=:log, title="",titleSize=12) where {N,T}
    mkpath(saveDirectory)
    listOfPaths = [pathType(0,[])]
    n = sizes(bspline, layers.subsampling, layers.n)
    q = [numScales(layers.shears[i],n[i]) for i=1:layers.m+1]
    push!(listOfPaths, [pathType(1,i) for i=1:q[1]-1]...)
    for path in listOfPaths
        pathIndex = (path.m==0) ? "" : path.Idxs[1]
        savefig(comparePathsChildren(thinResult, path, layers; outputSubsample=outputSubsample, names=names, colorScheme=colorScheme, scale=scale, title="Layer $(path.m+1) Path $(pathIndex)", titleSize=titleSize),joinpath(saveDirectory,"Layer$(path.m+1)Path$(pathIndex).pdf"))
    end
end

# TODO figure out which plotter is the more useful of the two
