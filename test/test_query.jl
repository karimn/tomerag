using Test
using TomeRAG: query, SourceRegistry, register_source!, Source, ChunkingConfig,
               DEFAULT_CONTENT_TYPES, MockEmbeddingBackend, HeuristicBackend,
               initialize_store, ingest!

@testset "query() end to end" begin
    db = tempname() * ".duckdb"
    src = Source(
        id = "iron",
        name = "Ironsworn",
        system = "PbtA",
        db_path = db,
        embedding_model = "mock",
        embedding_dim = 8,
        license = :cc_by,
        chunking = ChunkingConfig(min_tokens=1, max_tokens=200),
        content_types = DEFAULT_CONTENT_TYPES,
    )
    reg = SourceRegistry()
    register_source!(reg, src)
    initialize_store(src)

    md_path = tempname() * ".md"
    write(md_path, """
    # Moves
    ## Delve the Depths
    **When you delve the depths**, roll +wits.
    ## Secure an Advantage
    **When you secure an advantage**, roll +heart.
    """)
    ingest!(reg, "iron", md_path;
            doc_id="iron-core", document_type=:core_rules, format=:markdown,
            embed_backend=MockEmbeddingBackend(dim=8),
            classify_backend=HeuristicBackend())

    results = query(reg, "delve the depths";
                    source="iron", top_k=2,
                    embed_backend=MockEmbeddingBackend(dim=8))
    @test length(results) >= 1
    @test any(occursin("delve the depths", lowercase(r.chunk.text)) for r in results)

    # content_type filter
    moves = query(reg, "anything";
                  source="iron", content_type=:move, top_k=5,
                  embed_backend=MockEmbeddingBackend(dim=8))
    @test all(r.chunk.content_type == :move for r in moves)
end
