using Test
using TomeRAG
using HTTP
using JSON3

function _test_registry()
    db_path = tempname() * ".duckdb"
    src = Source(
        id = "test-src",
        name = "Test Source",
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

    md = tempname() * ".md"
    write(md, """
    # Moves
    ## Face Danger
    **When you attempt something risky**, roll +attr. On a 10+, you succeed.
    ## Swear an Iron Vow
    **When you swear upon iron**, roll +heart. On a miss, the vow is imperilled.
    # Lore
    ## The Iron World
    A world of blasted heaths, dark forests, and ancient ruins.
    """)
    ingest!(reg, "test-src", md;
            doc_id = "test-doc", document_type = :core_rules, format = :markdown,
            embed_backend = MockEmbeddingBackend(dim=8),
            classify_backend = HeuristicBackend())
    return reg
end

@testset "REST API" begin
    test_port = 18080
    reg = _test_registry()
    backend = MockEmbeddingBackend(dim=8)
    server = serve(reg; port=test_port, async=true, embed_backend=backend)
    sleep(1.0)
    base = "http://localhost:$test_port"

    try
        @testset "GET /health" begin
            r = HTTP.get("$base/health")
            @test r.status == 200
            body = JSON3.read(r.body, Dict{String,Any})
            @test body["status"] == "ok"
        end

        @testset "GET /sources" begin
            r = HTTP.get("$base/sources")
            @test r.status == 200
            body = JSON3.read(r.body, Vector{Dict{String,Any}})
            @test length(body) == 1
            @test body[1]["id"] == "test-src"
        end

        @testset "GET /sources/:id" begin
            r = HTTP.get("$base/sources/test-src")
            @test r.status == 200
            body = JSON3.read(r.body, Dict{String,Any})
            @test body["chunk_count"] >= 1

            r404 = HTTP.get("$base/sources/no-such-source"; status_exception=false)
            @test r404.status == 404
        end

        @testset "POST /query" begin
            payload = JSON3.write(Dict(
                "text"   => "iron vow swear",
                "source" => "test-src",
                "top_k"  => 2,
            ))
            r = HTTP.post("$base/query",
                          ["Content-Type" => "application/json"], payload)
            @test r.status == 200
            body = JSON3.read(r.body, Vector{Dict{String,Any}})
            @test length(body) >= 1
            @test haskey(body[1], "text")
            @test haskey(body[1], "score")
            @test haskey(body[1], "rank")

            # Missing source → 400
            bad = HTTP.post("$base/query",
                            ["Content-Type" => "application/json"],
                            JSON3.write(Dict("text" => "hello"));
                            status_exception=false)
            @test bad.status == 400
        end

        @testset "POST /filter" begin
            payload = JSON3.write(Dict(
                "source"       => "test-src",
                "top_k"        => 10,
            ))
            r = HTTP.post("$base/filter",
                          ["Content-Type" => "application/json"], payload)
            @test r.status == 200
            body = JSON3.read(r.body, Vector{Dict{String,Any}})
            @test length(body) >= 1
            @test haskey(body[1], "text")

            # Filter by content_type
            move_payload = JSON3.write(Dict(
                "source"       => "test-src",
                "content_type" => "move",
                "top_k"        => 10,
            ))
            r2 = HTTP.post("$base/filter",
                           ["Content-Type" => "application/json"], move_payload)
            @test r2.status == 200
            moves = JSON3.read(r2.body, Vector{Dict{String,Any}})
            @test length(moves) >= 1
            @test all(m["content_type"] == "move" for m in moves)
        end

    finally
        close(server)
    end
end
