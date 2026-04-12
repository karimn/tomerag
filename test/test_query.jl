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

@testset "query() hybrid mode (RRF)" begin
    db = tempname() * ".duckdb"
    src = Source(
        id = "hybrid-test",
        name = "Hybrid Test",
        system = "PbtA",
        db_path = db,
        embedding_model = "mock",
        embedding_dim = 8,
        license = :homebrew,
        chunking = ChunkingConfig(min_tokens=1, max_tokens=200),
        content_types = DEFAULT_CONTENT_TYPES,
    )
    reg = SourceRegistry()
    register_source!(reg, src)
    initialize_store(src)

    md_path = tempname() * ".md"
    write(md_path, """
    # Rules
    ## Iron Vow
    **When you swear an iron vow**, roll +heart.
    ## Face Danger
    **When you face danger**, roll +edge.
    """)
    ingest!(reg, "hybrid-test", md_path;
            doc_id = "hybrid-doc", document_type = :core_rules, format = :markdown,
            embed_backend = MockEmbeddingBackend(dim=8),
            classify_backend = HeuristicBackend())

    # Hybrid mode (default) should return results
    results = query(reg, "iron vow heart";
                    source = "hybrid-test", top_k = 2,
                    embed_backend = MockEmbeddingBackend(dim=8))
    @test length(results) >= 1
    @test results[1].rank == 1

    # Dense-only mode still works
    dense_results = query(reg, "iron vow heart";
                          source = "hybrid-test", top_k = 2, mode = :dense,
                          embed_backend = MockEmbeddingBackend(dim=8))
    @test length(dense_results) >= 1
end

using TomeRAG: filter_chunks, lookup, multi_query, get_context
# Note: the `using TomeRAG: name` syntax imports non-exported names too.
# Tests fail below because these functions don't exist yet (added in Step 3).

@testset "filter_chunks" begin
    db = tempname() * ".duckdb"
    src = Source(
        id = "filter-test",
        name = "Filter Test",
        system = "PbtA",
        db_path = db,
        embedding_model = "mock",
        embedding_dim = 8,
        license = :homebrew,
        chunking = ChunkingConfig(min_tokens=1, max_tokens=200),
        content_types = DEFAULT_CONTENT_TYPES,
    )
    reg = SourceRegistry()
    register_source!(reg, src)
    initialize_store(src)

    md_path = tempname() * ".md"
    write(md_path, """
    # Moves
    ## Iron Vow
    **When you swear an iron vow**, roll +heart.
    ## Face Danger
    **When you face danger**, roll +edge.
    # Lore
    ## The Iron World
    A world of darkness and danger.
    """)
    ingest!(reg, "filter-test", md_path;
            doc_id = "filter-doc", document_type = :core_rules, format = :markdown,
            embed_backend = MockEmbeddingBackend(dim=8),
            classify_backend = HeuristicBackend())

    all_chunks = filter_chunks(reg, "filter-test"; top_k=100)
    @test length(all_chunks) >= 3

    moves = filter_chunks(reg, "filter-test"; content_type=:move, top_k=100)
    @test all(c.content_type == :move for c in moves)
    @test length(moves) >= 1

    # Pagination
    page1 = filter_chunks(reg, "filter-test"; top_k=2, offset=0)
    page2 = filter_chunks(reg, "filter-test"; top_k=2, offset=2)
    @test length(page1) == 2
    @test (isempty(page2) || page1[1].id != page2[1].id)
end

@testset "lookup" begin
    db = tempname() * ".duckdb"
    src = Source(
        id = "lookup-test",
        name = "Lookup Test",
        system = "PbtA",
        db_path = db,
        embedding_model = "mock",
        embedding_dim = 8,
        license = :homebrew,
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
    ingest!(reg, "lookup-test", md_path;
            doc_id = "lookup-doc", document_type = :core_rules, format = :markdown,
            embed_backend = MockEmbeddingBackend(dim=8),
            classify_backend = HeuristicBackend())

    results = lookup(reg, "Delve the Depths"; source="lookup-test")
    @test length(results) >= 1
    @test occursin("delve", lowercase(results[1].chunk.text))
    @test results[1].rank == 1
end

@testset "multi_query" begin
    db = tempname() * ".duckdb"
    src = Source(
        id = "multi-test",
        name = "Multi Test",
        system = "PbtA",
        db_path = db,
        embedding_model = "mock",
        embedding_dim = 8,
        license = :homebrew,
        chunking = ChunkingConfig(min_tokens=1, max_tokens=200),
        content_types = DEFAULT_CONTENT_TYPES,
    )
    reg = SourceRegistry()
    register_source!(reg, src)
    initialize_store(src)

    md_path = tempname() * ".md"
    write(md_path, """
    # Rules
    ## Iron Vow
    **When you swear upon iron**, roll +heart. On a 10+, your vow is strong.
    ## Face Danger
    **When you face danger**, roll +edge. On a miss, pay the price.
    """)
    ingest!(reg, "multi-test", md_path;
            doc_id = "multi-doc", document_type = :core_rules, format = :markdown,
            embed_backend = MockEmbeddingBackend(dim=8),
            classify_backend = HeuristicBackend())

    backend = MockEmbeddingBackend(dim=8)
    results = multi_query(reg, [
        ("iron vow swear", (source="multi-test",)),
        ("face danger miss", (source="multi-test",)),
    ]; embed_backend=backend, top_k=5)

    @test length(results) >= 1
    @test results[1].rank == 1
    # Scores are normalized 0–1
    @test all(0.0f0 <= r.score <= 1.0f0 for r in results)
    # No duplicates
    ids = [r.chunk.id for r in results]
    @test length(ids) == length(unique(ids))
end

@testset "get_context" begin
    db = tempname() * ".duckdb"
    src = Source(
        id = "ctx-test",
        name = "Ctx Test",
        system = "PbtA",
        db_path = db,
        embedding_model = "mock",
        embedding_dim = 8,
        license = :homebrew,
        chunking = ChunkingConfig(min_tokens=1, max_tokens=200),
        content_types = DEFAULT_CONTENT_TYPES,
    )
    reg = SourceRegistry()
    register_source!(reg, src)
    initialize_store(src)

    md_path = tempname() * ".md"
    write(md_path, """
    # Rules
    ## Move A
    First move text here for context testing purposes.
    ## Move B
    Second move text here for context testing purposes.
    ## Move C
    Third move text here for context testing purposes.
    """)
    ingest!(reg, "ctx-test", md_path;
            doc_id = "ctx-doc", document_type = :core_rules, format = :markdown,
            embed_backend = MockEmbeddingBackend(dim=8),
            classify_backend = HeuristicBackend())

    all_chunks = filter_chunks(reg, "ctx-test"; top_k=10)
    @test length(all_chunks) >= 3

    # Get context around the middle chunk
    middle = all_chunks[2]
    ctx = get_context(reg, middle.id; source="ctx-test", before=1, after=1)
    @test length(ctx) >= 1
    @test any(c.id == middle.id for c in ctx)
    # Chunks are ordered by chunk_order
    orders = [c.chunk_order for c in ctx]
    @test orders == sort(orders)

    # Non-existent chunk returns empty
    empty_ctx = get_context(reg, "nonexistent-id"; source="ctx-test", before=1, after=1)
    @test isempty(empty_ctx)
end
