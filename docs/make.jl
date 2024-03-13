using SHTnsSpheres
using Documenter

DocMeta.setdocmeta!(SHTnsSpheres, :DocTestSetup, :(using SHTnsSpheres); recursive=true)

makedocs(;
    modules=[SHTnsSpheres],
    authors="The ClimFlows contributors",
    sitename="SHTnsSpheres.jl",
    format=Documenter.HTML(;
        canonical="https://ClimFlows.github.io/SHTnsSpheres.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ClimFlows/SHTnsSpheres.jl",
    devbranch="main",
)
