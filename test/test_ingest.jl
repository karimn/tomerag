using Test
using TomeRAG: ingest!, SourceRegistry, register_source!, Source, ChunkingConfig,
               DEFAULT_CONTENT_TYPES, MockEmbeddingBackend, HeuristicBackend,
               source_stats, initialize_store, filter_chunks,
               PageText, MockExtractionBackend

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

@testset "ingest! format=:pdf with MockExtractionBackend" begin
    db_path = tempname() * ".duckdb"
    src = Source(
        id = "pdf-ingest-test",
        name = "PDF Ingest Test",
        system = "PbtA",
        db_path = db_path,
        embedding_model = "mock",
        embedding_dim = 8,
        license = :homebrew,
        chunking = ChunkingConfig(min_tokens=1, max_tokens=200),
        content_types = DEFAULT_CONTENT_TYPES,
    )
    reg = SourceRegistry()
    register_source!(reg, src)
    initialize_store(src)

    pdf_path = tempname() * ".pdf"
    write(pdf_path, "fake pdf bytes — content ignored by mock")

    pages = [
        PageText(page_num=1, text="# Iron Vow\n**When you swear upon iron**, roll +heart. On a 10+, your vow is strong."),
        PageText(page_num=2, text="# Face Danger\n**When you face danger**, roll +edge. On a miss, pay the price."),
    ]
    extractor = MockExtractionBackend(pages)

    n = ingest!(reg, "pdf-ingest-test", pdf_path;
                doc_id             = "test-rules",
                document_type      = :core_rules,
                format             = :pdf,
                embed_backend      = MockEmbeddingBackend(dim=8),
                classify_backend   = HeuristicBackend(),
                extraction_backend = extractor)

    @test n >= 1
    chunks = filter_chunks(reg, "pdf-ingest-test"; top_k=20)
    @test !isempty(chunks)
    # Page numbers populated from markers
    @test any(c.page == "1" for c in chunks)
    @test any(c.page == "2" for c in chunks)
    # No page markers in chunk text
    @test all(!occursin(r"<!-- page \d+ -->", c.text) for c in chunks)
end

@testset "ingest! format=:auto detects PDF by extension" begin
    db_path = tempname() * ".duckdb"
    src = Source(
        id = "auto-pdf-test",
        name = "Auto Test",
        system = "PbtA",
        db_path = db_path,
        embedding_model = "mock",
        embedding_dim = 8,
        license = :homebrew,
        chunking = ChunkingConfig(min_tokens=1, max_tokens=200),
        content_types = DEFAULT_CONTENT_TYPES,
    )
    reg = SourceRegistry()
    register_source!(reg, src)
    initialize_store(src)

    pdf_path = tempname() * ".pdf"        # .pdf extension → auto-detect as :pdf
    write(pdf_path, "placeholder")

    pages = [PageText(page_num=1, text="# Content\nAuto-detected pdf content.")]
    extractor = MockExtractionBackend(pages)

    n = ingest!(reg, "auto-pdf-test", pdf_path;
                doc_id             = "auto-doc",
                document_type      = :core_rules,
                embed_backend      = MockEmbeddingBackend(dim=8),
                classify_backend   = HeuristicBackend(),
                extraction_backend = extractor)
    @test n >= 1
end

@testset "ingest! format=:auto detects markdown by extension" begin
    db_path = tempname() * ".duckdb"
    src = Source(
        id = "auto-md-test",
        name = "Auto MD Test",
        system = "PbtA",
        db_path = db_path,
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
    write(md_path, "# Rules\n**When you act**, roll +stat. On a 10+, succeed.")

    # No extraction_backend needed for markdown
    n = ingest!(reg, "auto-md-test", md_path;
                doc_id           = "md-doc",
                document_type    = :core_rules,
                embed_backend    = MockEmbeddingBackend(dim=8),
                classify_backend = HeuristicBackend())
    @test n >= 1
end

@testset "ingest! format=:pdf without extraction_backend raises error" begin
    reg = SourceRegistry()
    @test_throws ErrorException ingest!(reg, "any", "file.pdf";
                                        doc_id           = "x",
                                        document_type    = :core_rules,
                                        format           = :pdf,
                                        embed_backend    = MockEmbeddingBackend(dim=8),
                                        classify_backend = HeuristicBackend())
end
