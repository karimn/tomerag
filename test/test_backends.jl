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
using TomeRAG: ClaudeBackend, _build_batch_prompt

@testset "ClaudeBackend — constructs with required fields" begin
    b = ClaudeBackend(
        api_key       = "sk-fake",
        content_types = Set([:move, :mechanic, :lore]),
    )
    @test b.model == "claude-haiku-4-5-20251001"
    @test b.batch_size == 20
    @test :move in b.content_types
end

@testset "_build_batch_prompt — structure" begin
    b = ClaudeBackend(
        api_key       = "sk-fake",
        content_types = Set([:move, :mechanic, :lore]),
        system_hint   = "PbtA",
    )
    raws = [
        RawChunk(heading_path=["Chapter 3", "Iron Vow"],
                 text="**When you swear upon iron**, roll +heart.",
                 chunk_order=1),
        RawChunk(heading_path=["The World"],
                 text="The Ironlands are cold.",
                 chunk_order=2),
    ]
    prompt = _build_batch_prompt(b, raws)
    @test occursin("[1]", prompt)
    @test occursin("[2]", prompt)
    @test occursin("Iron Vow", prompt)
    @test occursin("move", prompt)        # content type listed
    @test occursin("mechanic", prompt)
    @test occursin("content_type", prompt)
    @test occursin("move_trigger", prompt)
end

@testset "ClaudeBackend — classify_batch live (requires API key + TOMERAG_LIVE_TESTS=1)" begin
    if get(ENV, "TOMERAG_LIVE_TESTS", "0") != "1" || !haskey(ENV, "ANTHROPIC_API_KEY")
        @test_skip "Set TOMERAG_LIVE_TESTS=1 and ANTHROPIC_API_KEY to run"
    else
        b = ClaudeBackend(
            content_types = Set([:move, :mechanic, :lore, :stat_block, :table,
                                 :gm_guidance, :flavor, :procedure, :boxed_text]),
            system_hint   = "PbtA",
        )
        raws = [
            RawChunk(heading_path=["Moves", "Iron Vow"],
                     text="**When you swear upon iron**, roll +heart. On a 10+, your vow is strong.",
                     chunk_order=1),
            RawChunk(heading_path=["Bestiary", "Troll"],
                     text="HP 30, Armor 1, Attack: Club 2d6. Regenerates 2 HP per round.",
                     chunk_order=2),
            RawChunk(heading_path=["The World", "History"],
                     text="The Ironlands were settled generations ago by refugees fleeing the Old World.",
                     chunk_order=3),
        ]
        results = classify_batch(b, raws)
        @test length(results) == 3
        for r in results
            @test r.content_type in b.content_types
        end
        # Move chunk should be classified as :move
        @test results[1].content_type == :move
    end
end

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

using TomeRAG: CostEstimate, forecast_cost

@testset "forecast_cost live (requires ANTHROPIC_API_KEY + TOMERAG_LIVE_TESTS=1)" begin
    if get(ENV, "TOMERAG_LIVE_TESTS", "0") != "1" || !haskey(ENV, "ANTHROPIC_API_KEY")
        @test_skip "Set TOMERAG_LIVE_TESTS=1 and ANTHROPIC_API_KEY to run"
    else
        b = ClaudeBackend(
            content_types = Set([:move, :mechanic, :lore]),
            system_hint   = "PbtA",
        )
        raws = [
            RawChunk(heading_path=["Moves", "Iron Vow"],
                     text="**When you swear upon iron**, roll +heart. On a 10+, your vow is strong.",
                     chunk_order=1),
            RawChunk(heading_path=["The World"],
                     text="The Ironlands are cold and ancient.",
                     chunk_order=2),
            RawChunk(heading_path=["Bestiary", "Troll"],
                     text="HP 30, Armor 1, Attack: Club 2d6.",
                     chunk_order=3),
        ]
        est = forecast_cost(b, raws)
        @test est isa CostEstimate
        @test est.model == b.model
        @test est.n_chunks == 3
        @test est.n_batches == 1            # 3 chunks < batch_size=20
        @test est.input_tokens > 0
        @test est.output_tokens == 3 * 50   # 50 tokens/chunk estimate
        @test est.total_cost_usd > 0.0f0
    end
end
