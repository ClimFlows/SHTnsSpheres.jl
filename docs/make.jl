using SHTnsSphere
using Documenter

DocMeta.setdocmeta!(SHTnsSphere, :DocTestSetup, :(using SHTnsSphere); recursive=true)

makedocs(;
    modules=[SHTnsSphere],
    authors="The ClimFlows contributors",
    sitename="SHTnsSphere.jl",
    format=Documenter.HTML(;
        canonical="https://ClimFlows.github.io/SHTnsSphere.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ClimFlows/SHTnsSphere.jl",
    devbranch="main",
)
