using ScatteringTransform, Plots, Colors
darker_blue = RGB(0.251, 0.388, 0.847)
lighter_blue = RGB(0.4, 0.51, 0.878)
darker_purple = RGB(0.584, 0.345, 0.698)
lighter_purple = RGB(0.667, 0.475, 0.757)
darker_green = RGB(0.22, 0.596, 0.149)
lighter_green = RGB(0.376, 0.678, 0.318)
darker_red = RGB(0.796, 0.235, 0.2)
lighter_red = RGB(0.835, 0.388, 0.361)
using ContinuousWavelets, LinearAlgebra
st = scatteringTransform((3052, 1, 1); cw=dog2, p=1, averagingLength=-1, β=1.0)
w1, w2, w3 = getWavelets(st, spaceDomain=true)
w = w1 ./ norm.(eachcol(w1), Inf)'
plot(circshift(w, (1000, 0))[400:1600, [2, 20, end]], c=[darker_purple darker_green darker_red], linewidth=9, legend=false, xaxis=false, yaxis=false, xticks=false, yticks=false, background_color=false)
savefig("src/assets/logo.svg")
savefig("src/assets/logo.png")
