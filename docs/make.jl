using Documenter, ScatteringTransform, ScatteringPlots

ENV["PLOTS_TEST"] = "true"
ENV["GKSwstype"] = "100"
ENV["LINES"] = "9"
ENV["COLUMNS"] = "60"

makedocs(;
    modules=[ScatteringTransform, ScatteringPlots],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
        "Mathematical Description" => "math.md",
        "Scattering Transform" => [
            "scatteringTransform type" => "struct.md",
            "ScatteredOut type" => "out.md",
            "Subsampling Operators" => "subsampling.md",
            "Path Operations" => "pathLocs.md",
            "Utilities" => "utils.md",
        ],
        "Scattering Plots" => [
            "Plots" => "plots.md",
        ],
    ],
    sitename="ScatteringTransform.jl",
    authors="David Weber",
    clean=true
)

deploydocs(;
    repo="github.com/dsweber2/ScatteringTransform.jl.git"
)
