########################################################
########################################################
# PLOTTING
########################################################
########################################################
"""
plotShattered(sheared::Array{Float64,3}, layers::layeredTransform; m=2, p=2, scale=:log, saveTo="",title="Shattering Transform Coefficients")

Given a shattering transform and a layeredTransform of at least depth 2, it plots the magnitude in each coefficient .
Given a classifier and a layeredTransform, it plots
"""
function plotShattered(output::Array{Float64,3}, layers::layeredTransform; m=2, p=2, scale=:log, saveTo="", title="Shattering Transform Coefficients", aspect_ratio=1)
  average = [sqrt(norm(output[:,:,i], 2)) for i=1:size(output,3)]
  xNames = Array{Int64}(layers.shears[m].shearletIdxs[1:end-1,:])
  xNames = ["$(xNames[i,:])  " for i=1:size(xNames,1)]
  n = size(layers.shears[m].shearletIdxs,1)-1
  yNames = Array{Int64}(layers.shears[m-1].shearletIdxs[1:end-1,:])
  yNames = ["$(yNames[i,:])  " for i=1:size(yNames,1)]
  k = size(layers.shears[m-1].shearletIdxs,1)-1
  # pyplot()
  # SIZE OF TRANSFORM
  if scale==:log
    plt = plot(xNames,yNames, reshape(log(average), (n,k)), colorbar_title="log $p norm", linetype=:heatmap, title="$title\n[cone, scale, shearing]",xlabel="layer 2 indices", ylabel="layer 1 indices")
  else
    plt = plot(xNames,yNames, reshape(average, (n,k)), colorbar_title="$p norm", linetype=:heatmap, title="$(title)\n[cone, scale, shearing]", xlabel="layer 2 indices", ylabel="layer 1 indices")
  end
  if saveTo!=""
    savefig(saveTo)
  end
  return plt
end

"""
plotShattered(sheared::shearedArray, layers::layeredTransform; m=2, p=2, scale=:log, saveTo="",title="Shattering Transform Coefficients")

Given a shattering transform and a layeredTransform of at least depth 2, it plots the magnitude in each coefficient .
Given a classifier and a layeredTransform, it plots
"""
function plotShattered(sheared::shearedArray, layers::layeredTransform; m=2, p=2, scale=:log, saveTo="",title="Shattering Transform Coefficients", aspect_ratio=1)
  plotShattered(sheared.output[m+1], layers, m=m, p=p, scale=scale, saveTo=saveTo, title=title,aspect_ratio=aspect_ratio)
end


"""
    plotCoordinate(index::Vector{Array{Int64,1}}, m::Array{Int64}, layers::layeredTransform; fun=logabs, saveTo="")

given an index as listed in layers (starting with the second first), and a depth m, plot the corresponding shearlets, or if also given a shearedArray, plot the actual output for that coordinate. fun should be either abs, real, or imag
"""
function plotCoordinate(index::Vector{Array{Int64,1}}, m::Array{Int64}, layers::layeredTransform; fun=logabs, saveTo="")
  i=1
  for i = 1:size(layers.shears[m[1]].shearletIdxs,1)
    if layers.shears[m[1]].shearletIdxs[i,:]==index[1]
      break
    end
  end
  j=1
  for j = 1:size(layers.shears[m[2]].shearletIdxs,1)
    if layers.shears[m[2]].shearletIdxs[j,:]==index[2]
      break
    end
  end
  plt = plot(plot(fun(layers.shears[m[1]].shearlets[:,:,i]), linetype=:heatmap, xticks=[], yticks=[],title="First [cone, scale, shearing]=$index"), plot(fun(layers.shears[m[2]].shearlets[:,:,j]), linetype=:heatmap, xticks=[], yticks=[],title="Second [cone, scale, shearing]=$index"), layout=(2,1))
  if saveTo!=""
    savefig(saveTo)
  end
  return plt
end

function plotCoordinate(index::Vector{Array{Int64,1}}, m::Int64, layers::layeredTransform, sheared::shearedArray; fun=abs, title="", scale=:none, saveTo="")
  # i is the index from the first to the last
  i=[1 for j=1:m]
  for j=1:m
    for i[j] = 1:size(layers.shears[j].shearletIdxs,1)
      if layers.shears[j].shearletIdxs[i[j],:]==index[j]
        break
      end
    end
  end
  i[end]=i[end]+1
  netIndex = sum([(i[j]-1)*numberSkipped(m,j,layers) for j=1:m])
  plt = plot(fun(sheared.output[m+1][:,:,netIndex]), linetype=:heatmap, xticks=[], yticks=[],title="$(title)\nm=$m, [cone, scale, shearing]=$(index[2]) $(index[1])")
  if saveTo!=""
    savefig(saveTo)
  end
  return plt
  # return plot(, linetype=:heatmap)
end

# TODO: A version of plotCoordinate that works for a list of indices
# playing around with a way to automatically get square layouts
# r=3
# r1 = Int64(round(sqrt(r)))
# Int64(ceil(r/r1))
