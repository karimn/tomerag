using Test
using TomeRAG

@testset "integration: ingest + query against fixture" begin
    fixture = joinpath(@__DIR__, "fixtures", "ironsworn_sample.md")
    db = tempname() * ".duckdb"

    src = Source(
        id = "iron",
        name = "Ironsworn",
        system = "PbtA",
        db_path = db,
        embedding_model = "mock",
        embedding_dim = 16,
        license = :cc_by,
        chunking = ChunkingConfig(min_tokens=5, max_tokens=300),
        content_types = PBTA_CONTENT_TYPES,
    )
    reg = SourceRegistry()
    register_source!(reg, src)
    initialize_store(src)

    n = ingest!(reg, "iron", fixture;
                doc_id = "ironsworn-sample",
                document_type = :core_rules,
                format = :markdown,
                embed_backend = MockEmbeddingBackend(dim=16),
                classify_backend = HeuristicBackend())
    @test n >= 4

    stats = source_stats(src)
    @test stats.chunk_count == n

    results = query(reg, "delve the depths";
                    source = "iron", top_k = 3,
                    embed_backend = MockEmbeddingBackend(dim=16))
    @test length(results) >= 1
    @test results[1].rank == 1
    @test occursin("delve the depths", lowercase(results[1].chunk.text))

    # Heuristic should tag moves correctly
    move_results = query(reg, "anything";
                         source = "iron", content_type = :move, top_k = 10,
                         embed_backend = MockEmbeddingBackend(dim=16))
    @test length(move_results) >= 2
    @test all(r.chunk.content_type == :move for r in move_results)

    # Re-ingest is a no-op (dedup)
    n2 = ingest!(reg, "iron", fixture;
                 doc_id = "ironsworn-sample",
                 document_type = :core_rules,
                 format = :markdown,
                 embed_backend = MockEmbeddingBackend(dim=16),
                 classify_backend = HeuristicBackend())
    @test n2 == 0
    @test source_stats(src).chunk_count == n
end
