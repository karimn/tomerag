using Test
using TomeRAG: ingest!, SourceRegistry, register_source!, Source, ChunkingConfig,
               DEFAULT_CONTENT_TYPES, MockEmbeddingBackend, HeuristicBackend,
               source_stats, initialize_store

@testset "ingest! markdown end-to-end with mocks" begin
    db = tempname() * ".duckdb"
    src = Source(
        id = "iron",
        name = "Ironsworn",
        system = "PbtA",
        db_path = db,
        embedding_model = "mock",
        embedding_dim = 8,
        license = :cc_by,
        chunking = ChunkingConfig(min_tokens=1, max_tokens=200, overlap_tokens=0),
        content_types = DEFAULT_CONTENT_TYPES,
    )
    reg = SourceRegistry()
    register_source!(reg, src)
    initialize_store(src)

    md_path = tempname() * ".md"
    write(md_path, """
    # Moves

    ## Delve the Depths

    **When you delve the depths**, roll +wits. On a 10+, choose two.

    ## Secure an Advantage

    **When you secure an advantage**, roll +heart.

    ## Aid Your Ally

    **When you aid your ally**, roll +heart. On a hit, they take +1 on their move.
    """)

    n = ingest!(reg, "iron", md_path;
                doc_id = "ironsworn-core",
                document_type = :core_rules,
                format = :markdown,
                embed_backend = MockEmbeddingBackend(dim=8),
                classify_backend = HeuristicBackend())
    @test n == 3
    @test source_stats(src).chunk_count == 3

    # Idempotent: re-ingest inserts 0
    n2 = ingest!(reg, "iron", md_path;
                 doc_id = "ironsworn-core",
                 document_type = :core_rules,
                 format = :markdown,
                 embed_backend = MockEmbeddingBackend(dim=8),
                 classify_backend = HeuristicBackend())
    @test n2 == 0
end
