using Test
using TomeRAG: EmbeddingBackend, ClassifyBackend, MockEmbeddingBackend, MockClassifyBackend,
               embed, classify

@testset "mock embedding backend" begin
    b = MockEmbeddingBackend(dim=4)
    v = embed(b, "hello world")
    @test length(v) == 4
    @test eltype(v) == Float32

    # deterministic: same text -> same vector
    @test embed(b, "foo") == embed(b, "foo")
    @test embed(b, "foo") != embed(b, "bar")

    # batch
    vs = embed(b, ["a", "b", "c"])
    @test length(vs) == 3
    @test all(length(v) == 4 for v in vs)
end

@testset "mock classify backend" begin
    c = MockClassifyBackend(content_type=:mechanic, tags=["x"])
    out = classify(c, text="anything", heading_path=["a"])
    @test out.content_type == :mechanic
    @test out.tags == ["x"]
    @test out.move_trigger === nothing
end
