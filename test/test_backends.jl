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

using TomeRAG: classify_batch, RawChunk

@testset "classify_batch default — same result as classify loop" begin
    b = HeuristicBackend()
    raws = [
        RawChunk(heading_path=["Moves", "Iron Vow"],
                 text="**When you swear upon iron**, roll +heart. On a 10+, your vow is strong.",
                 chunk_order=1),
        RawChunk(heading_path=["Bestiary", "Ironclad"],
                 text="HP 15, Armor 2, Attack: Blade 1d6.",
                 chunk_order=2),
        RawChunk(heading_path=["The World", "Geography"],
                 text="The Ironlands stretch far to the north, cold and unforgiving.",
                 chunk_order=3),
    ]

    batch_results  = classify_batch(b, raws)
    single_results = [classify(b; text=r.text, heading_path=r.heading_path) for r in raws]

    @test length(batch_results) == 3
    for i in 1:3
        @test batch_results[i].content_type == single_results[i].content_type
        @test batch_results[i].move_trigger == single_results[i].move_trigger
    end
end
