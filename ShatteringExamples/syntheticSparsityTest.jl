using Shearlab, FFTW, Plots
function reorderCoeff(A::AbstractArray{T,2}) where {T}
    if size(A, 1) == 1
        includingZeros = reshape(sort(abs.(A), dims =
                length(size(A)), rev = true), (length(A,)))
    else
        includingZeros = sort(abs.(A), dims = length(size(A)), rev = true)
    end
    firstZero = findfirst(includingZeros .== 0)
    if typeof(firstZero) <: Nothing
        return includingZeros
    else
        return includingZeros[1:firstZero]
    end
end
reorderCoeff(A::AbstractArray{T,1}) where {T} = sort(abs.(A), rev = true)


N = 32
xlocs = range(-1, 1, length = 32)
rot(θ) = [cos(θ) -sin(θ); sin(θ) cos(θ)]
x = 1;
y = 0;
θ = π / 3;
σx = 0.4;
σy = 0.1;
f(xloc, yloc, σx, σy, θ) = [exp.(-1 / 2 * [x, y]' * rot(θ)' * ([σx^2 0; 0 σy^2])^(-1) * rot(θ) * [x, y]) for x in xloc, y in yloc]

gaussianExample = f(xlocs, xlocs, 0.6, 0.3, π / 6)
heatmap(gaussianExample)

shears = Shearlab.getshearletsystem2D(n, n, 4, typeBecomes = Float32)
size(shears.shearlets)
sheared = Shearlab.sheardec2D(Float32.(gaussianExample), shears)[:, :, 1:(end-1)]

mutilated = copy(gaussianExample);
nZeros = floor(Int, 12 / 32 * n);
mutilated[1:nZeros, :] = zeros(nZeros, size(mutilated, 2));
heatmap(mutilated)


mutsheared = Shearlab.sheardec2D(Float32.(mutilated), shears)[:, :, 1:(end-1)]

reorderedShear = reorderCoeff(reshape(sheared, :))
reorderedMutShear = reorderCoeff(reshape(mutsheared, :))
pyplot()
plot(reorderedShear)
plot(log.(1:length(reorderedShear)), max.(-25, log.([reorderedShear reorderedMutShear])), labels = ["original", "mutilated"])
savefig("mutilated_Gaussian_Decay_Shearlets_32_4Scales.pdf")
