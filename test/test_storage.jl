using Test
using TomeRAG: initialize_store, insert_chunks, Chunk, Source, ChunkingConfig,
               DEFAULT_CONTENT_TYPES, source_stats, similarity_search, bm25_search

function _mk_source(path; dim=4)
    Source(
        id = "coriolis",
        name = "Coriolis",
        system = "YZE",
        db_path = path,
        embedding_model = "mock",
        embedding_dim = dim,
        license = :homebrew,
        chunking = ChunkingConfig(),
        content_types = DEFAULT_CONTENT_TYPES,
    )
end

function _mk_chunk(id, text; dim=4)
    Chunk(
        id = id, source_id = "coriolis",
        doc_id = "doc", doc_path = "/tmp/doc.md",
        text = text, embedding = Float32[0.1, 0.2, 0.3, 0.4][1:dim],
        embedding_model = "mock", token_count = length(split(text)),
        content_hash = "h_$id",
        document_type = :core_rules, system = "YZE", edition = "1e",
        page = "", heading_path = ["Ch"], chunk_order = 0, parent_id = nothing,
        content_type = :mechanic, tags = String[],
        move_trigger = nothing, scene_type = nothing,
        encounter_key = nothing, npc_name = nothing,
        license = :homebrew,
    )
end

@testset "storage initialize + insert" begin
    path = tempname() * ".duckdb"
    src = _mk_source(path)
    initialize_store(src)

    chunks = [_mk_chunk("a", "hello world"), _mk_chunk("b", "goodbye world")]
    n = insert_chunks(src, chunks)
    @test n == 2

    stats = source_stats(src)
    @test stats.chunk_count == 2
    @test stats.embedding_model == "mock"
    @test stats.embedding_dim == 4
end

@testset "storage dedup scoped to (doc_id, content_hash)" begin
    path = tempname() * ".duckdb"
    src = _mk_source(path)
    initialize_store(src)

    c = _mk_chunk("a", "hello world")
    @test insert_chunks(src, [c]) == 1
    @test insert_chunks(src, [c]) == 0        # same content_hash → skipped
    @test source_stats(src).chunk_count == 1
end

@testset "similarity_search" begin
    path = tempname() * ".duckdb"
    src = _mk_source(path)
    initialize_store(src)

    c1 = _mk_chunk("a", "blight corruption spreads")
    c2 = _mk_chunk("b", "delve the deeps")
    # Override embeddings with known distinct vectors
    c1 = Chunk(c1; embedding = Float32[1.0, 0.0, 0.0, 0.0])
    c2 = Chunk(c2; embedding = Float32[0.0, 1.0, 0.0, 0.0])
    insert_chunks(src, [c1, c2])

    # Query vector closer to c1
    q = Float32[0.9, 0.1, 0.0, 0.0]
    results = similarity_search(src, q; top_k=2)
    @test length(results) == 2
    @test results[1].chunk.id == "a"
    @test results[1].rank == 1
    @test 0 <= results[1].score <= 1
end

@testset "bm25_search" begin
    path = tempname() * ".duckdb"
    src = _mk_source(path)
    initialize_store(src)

    # Insert chunks with distinct text
    c1 = _mk_chunk("bm1", "iron vow momentum move roll")
    c2 = _mk_chunk("bm2", "delve the dungeon depths explore")
    c3 = _mk_chunk("bm3", "blight corruption spreads darkness")
    insert_chunks(src, [c1, c2, c3])

    results = bm25_search(src, "momentum iron vow"; top_k=3)
    @test length(results) >= 1
    # First result should be the momentum/iron vow chunk
    top_chunk, top_score = results[1]
    @test top_chunk.id == "bm1"
    @test top_score > 0.0f0

    # Filter by content_type returns only matching chunks
    c4 = _mk_chunk("bm4", "iron vow move lore")
    c4 = Chunk(c4; content_type = :lore)
    insert_chunks(src, [c4])
    lore_results = bm25_search(src, "iron vow";
                               top_k=5,
                               filters=Dict{String,Any}("content_type" => "lore"))
    @test all(r[1].content_type == :lore for r in lore_results)
end
