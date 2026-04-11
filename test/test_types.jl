using Test
using TomeRAG: Chunk, QueryResult, ChunkingConfig

@testset "types" begin
    cfg = ChunkingConfig()
    @test cfg.min_tokens == 100
    @test cfg.max_tokens == 800
    @test cfg.overlap_tokens == 50
    @test cfg.overflow == :paragraph

    c = Chunk(
        id              = "abc",
        source_id       = "coriolis",
        doc_id          = "core-1e",
        doc_path        = "/tmp/x.md",
        text            = "hello",
        embedding       = Float32[0.1, 0.2],
        embedding_model = "mock",
        token_count     = 1,
        content_hash    = "h",
        document_type   = :core_rules,
        system          = "YZE",
        edition         = "1e",
        page            = "1",
        heading_path    = ["Ch", "Sec"],
        chunk_order     = 0,
        parent_id       = nothing,
        content_type    = :mechanic,
        tags            = ["x"],
        move_trigger    = nothing,
        scene_type      = nothing,
        encounter_key   = nothing,
        npc_name        = nothing,
        license         = :homebrew,
    )
    @test c.id == "abc"
    @test c.parent_id === nothing
    @test length(c.embedding) == 2

    r = QueryResult(c, 0.9f0, 1)
    @test r.score == 0.9f0
    @test r.rank == 1
end
