using Test
using TomeRAG: EmbeddingBackend, ClassifyBackend, MockEmbeddingBackend, MockClassifyBackend,
               embed, classify, VisionBackend
using TomeRAG: _get_anthropic_key

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

@testset "_get_anthropic_key — reads from ANTHROPIC_API_KEY env var" begin
    withenv("ANTHROPIC_API_KEY" => "sk-test-from-env") do
        @test _get_anthropic_key() == "sk-test-from-env"
    end
end

@testset "_get_anthropic_key — errors with helpful message when unset" begin
    withenv("ANTHROPIC_API_KEY" => nothing) do
        err = try
            _get_anthropic_key()
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("set_preferences!", err.msg)
    end
end

@testset "VisionBackend constructs with explicit api_key" begin
    b = VisionBackend(api_key="sk-fake-key")
    @test b.api_key == "sk-fake-key"
end
