ENV["PLOTS_TEST"] = "true"
ENV["GKSwstype"] = "100"
ENV["LINES"] = "9"
ENV["COLUMNS"] = "60"
using Documenter, ScatteringTransform, ScatteringPlots

makedocs(
    sitename = "ScatteringTransform.jl",
    format = Documenter.HTML(),
    authors="David Weber, Naoki Saito",
    clean=true,
    # This ignores the ContinuousWavelets warnings during doctests
    doctestfilters = [
        r"┌ Warning:.*",
        r"│.*",
        r"└ @ ContinuousWavelets.*"
    ],
    pages = Any[
         "Home" => "index.md",
         "Scattering Transform" => Any[
            "scatteringTransform type" => "struct.md",
            "ScatteredOut type" => "out.md",
            "Subsampling Operators" => "subsampling.md",
            "Path Operations" => "pathLocs.md",
            "Utilities" => "utils.md",
            "Plotting Utilities" => "plots.md",
        ],
    ],
)

deploydocs(;
    repo="github.com/BoundaryValueProblems/ScatteringTransform.jl.git"
)
