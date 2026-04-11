using Test
using TomeRAG

@testset "TomeRAG.jl" begin
    @test isdefined(Main, :TomeRAG) || isdefined(@__MODULE__, :TomeRAG)
end
