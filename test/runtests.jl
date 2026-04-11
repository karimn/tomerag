using Test
using TomeRAG

@testset "TomeRAG.jl" begin
    include("test_content_types.jl")
    include("test_types.jl")
end
