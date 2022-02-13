using ScatteringTransform
sourceFolder = "/home/dsweber/Dropbox/matlab/sonarProject/data/sf"
destFolder = "/home/dsweber/Dropbox/matlab/sonarProject/scatteredData/sf"
loadThis = loadSyntheticMatFile
nonlinear = abs;
subsam = bspline;
stType = "full";
(fullMatrix, rando) = loadSyntheticMatFile("$(sourceFolder)/1000/data00.mat")
layers = stParallel(3, fullMatrix[1, :], [8, 4, 2, 1])
transformFolder(sourceFolder, destFolder, layers, false)
# using JLD
# results = load("/home/dsweber/Dropbox/matlab/sonarProject/scatteredData/sf/1000/data110.mat.jld")
# awe = results["result"]
# awe
