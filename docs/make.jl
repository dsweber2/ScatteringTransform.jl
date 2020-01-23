using Documenter, ScatteringTransform

makedocs(;
    modules=[ScatteringTransform],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/dsweber2/ScatteringTransform.jl/blob/{commit}{path}#L{line}",
    sitename="ScatteringTransform.jl",
    authors="dsweber2",
    assets=String[],
)

deploydocs(;
    repo="github.com/dsweber2/ScatteringTransform.jl",
)
