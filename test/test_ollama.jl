using Test
using TomeRAG: OllamaBackend, embed

@testset "ollama backend construction" begin
    b = OllamaBackend(model="nomic-embed-text", base_url="http://localhost:11434", dim=768)
    @test b.model == "nomic-embed-text"
    @test b.dim == 768
    @test b.base_url == "http://localhost:11434"
    @test b.batch_size == 32
end

if get(ENV, "TOMERAG_TEST_OLLAMA", "") == "1"
    @testset "ollama live embed" begin
        b = OllamaBackend(model="nomic-embed-text", dim=768)
        v = embed(b, "hello")
        @test length(v) == 768
        @test eltype(v) == Float32

        vs = embed(b, ["alpha", "beta", "gamma"])
        @test length(vs) == 3
        @test all(length(x) == 768 for x in vs)
    end
end
