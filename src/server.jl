using Oxygen
using HTTP
using JSON3
using DBInterface

"""
    serve(registry; port=8080, async=false, embed_backend=nothing)

Start the TomeRAG REST API server.
- `async=true` returns a closeable handle (use `close(server)` to stop).
- `embed_backend` is required for endpoints that embed text (/query, /lookup, /multi_query).
  Pass a `MockEmbeddingBackend` for tests or an `OllamaBackend` for production.
"""
function serve(registry::SourceRegistry;
               port::Int=8080,
               async::Bool=false,
               embed_backend::Union{EmbeddingBackend,Nothing}=nothing)
    _setup_routes!(registry, embed_backend)
    return Oxygen.serve(; host="127.0.0.1", port=port, async=async)
end

function _setup_routes!(registry::SourceRegistry,
                        embed_backend::Union{EmbeddingBackend,Nothing})

    @get "/health" function(req::HTTP.Request)
        return Dict("status" => "ok")
    end

    @get "/sources" function(req::HTTP.Request)
        return [_source_info(src) for src in values(registry.sources)]
    end

    @get "/sources/{id}" function(req::HTTP.Request, id::String)
        if !haskey(registry.sources, id)
            return _error(404, "source not found: $id")
        end
        src = get_source(registry, id)
        stats = source_stats(src)
        return merge(_source_info(src), Dict("chunk_count" => stats.chunk_count))
    end

    @post "/query" function(req::HTTP.Request)
        isnothing(embed_backend) && return _error(501, "no embed_backend configured")
        body = try JSON3.read(req.body, Dict{String,Any}) catch; return _error(400, "invalid JSON") end
        haskey(body, "text")   || return _error(400, "missing field: text")
        haskey(body, "source") || return _error(400, "missing field: source")
        source = body["source"]
        haskey(registry.sources, source) || return _error(404, "source not found: $source")
        top_k           = Int(get(body, "top_k", 5))
        score_threshold = Float32(get(body, "score_threshold", 0.0))
        content_type    = haskey(body, "content_type")  ? Symbol(body["content_type"])  : nothing
        document_type   = haskey(body, "document_type") ? Symbol(body["document_type"]) : nothing
        mode            = haskey(body, "mode")           ? Symbol(body["mode"])          : :hybrid
        results = query(registry, String(body["text"]);
                        source=source, content_type=content_type,
                        document_type=document_type, top_k=top_k,
                        score_threshold=score_threshold,
                        embed_backend=embed_backend, mode=mode)
        return [_result_dict(r) for r in results]
    end

    @post "/filter" function(req::HTTP.Request)
        body = try JSON3.read(req.body, Dict{String,Any}) catch; return _error(400, "invalid JSON") end
        haskey(body, "source") || return _error(400, "missing field: source")
        source = body["source"]
        haskey(registry.sources, source) || return _error(404, "source not found: $source")
        content_type  = haskey(body, "content_type")  ? Symbol(body["content_type"])  : nothing
        document_type = haskey(body, "document_type") ? Symbol(body["document_type"]) : nothing
        top_k  = Int(get(body, "top_k", 100))
        offset = Int(get(body, "offset", 0))
        chunks = filter_chunks(registry, source;
                               content_type=content_type, document_type=document_type,
                               top_k=top_k, offset=offset)
        return [_chunk_dict(c) for c in chunks]
    end

    @get "/lookup" function(req::HTTP.Request)
        isnothing(embed_backend) && return _error(501, "no embed_backend configured")
        params = HTTP.queryparams(HTTP.URI(req.target))
        haskey(params, "name")   || return _error(400, "missing query param: name")
        haskey(params, "source") || return _error(400, "missing query param: source")
        source = params["source"]
        haskey(registry.sources, source) || return _error(404, "source not found: $source")
        content_type = haskey(params, "content_type") ? Symbol(params["content_type"]) : nothing
        results = lookup(registry, params["name"];
                         source=source, content_type=content_type)
        return [_result_dict(r) for r in results]
    end

    @post "/multi_query" function(req::HTTP.Request)
        isnothing(embed_backend) && return _error(501, "no embed_backend configured")
        body = try JSON3.read(req.body, Dict{String,Any}) catch; return _error(400, "invalid JSON") end
        haskey(body, "queries") || return _error(400, "missing field: queries")
        top_k = Int(get(body, "top_k", 10))
        queries = Tuple{String,NamedTuple}[]
        for q in body["queries"]
            haskey(q, "text")   || return _error(400, "each query needs 'text'")
            haskey(q, "source") || return _error(400, "each query needs 'source'")
            source = String(q["source"])
            haskey(registry.sources, source) || return _error(404, "source not found: $source")
            kwargs = (source=source,)
            push!(queries, (String(q["text"]), kwargs))
        end
        results = multi_query(registry, queries;
                              embed_backend=embed_backend, top_k=top_k)
        return [_result_dict(r) for r in results]
    end

    @get "/chunk/{id}" function(req::HTTP.Request, id::String)
        for src in values(registry.sources)
            db = DBInterface.connect(DuckDB.DB, src.db_path)
            try
                for row in DuckDB.execute(db,
                        """SELECT id, source_id, doc_id, doc_path, text,
                                  embedding::FLOAT[] AS embedding,
                                  embedding_model, token_count, content_hash,
                                  document_type, system, edition, page,
                                  heading_path, chunk_order, parent_id,
                                  content_type, tags, move_trigger,
                                  scene_type, encounter_key, npc_name, license
                           FROM chunks WHERE id = ? LIMIT 1""",
                        (id,))
                    return _chunk_dict(_row_to_chunk(row))
                end
            finally
                DBInterface.close!(db)
            end
        end
        return _error(404, "chunk not found: $id")
    end

    @get "/chunk/{id}/context" function(req::HTTP.Request, id::String)
        params = HTTP.queryparams(HTTP.URI(req.target))
        haskey(params, "source") || return _error(400, "missing query param: source")
        source = params["source"]
        haskey(registry.sources, source) || return _error(404, "source not found: $source")
        before = haskey(params, "before") ? parse(Int, params["before"]) : 1
        after  = haskey(params, "after")  ? parse(Int, params["after"])  : 1
        chunks = get_context(registry, id; source=source, before=before, after=after)
        return [_chunk_dict(c) for c in chunks]
    end
end

# ── Helpers ───────────────────────────────────────────────────────────────────

function _error(status::Int, msg::String)
    return HTTP.Response(status,
        ["Content-Type" => "application/json"],
        JSON3.write(Dict("error" => msg)))
end

function _source_info(src::Source)
    return Dict(
        "id"              => src.id,
        "name"            => src.name,
        "system"          => src.system,
        "embedding_model" => src.embedding_model,
        "embedding_dim"   => src.embedding_dim,
        "license"         => String(src.license),
    )
end

# Used by /query, /lookup, /multi_query routes (Task 6+)
function _chunk_dict(c::Chunk)
    return Dict(
        "id"            => c.id,
        "source_id"     => c.source_id,
        "doc_id"        => c.doc_id,
        "text"          => c.text,
        "content_type"  => String(c.content_type),
        "document_type" => String(c.document_type),
        "heading_path"  => c.heading_path,
        "page"          => c.page,
        "tags"          => c.tags,
        "move_trigger"  => c.move_trigger,
        "chunk_order"   => c.chunk_order,
    )
end

function _result_dict(r::QueryResult)
    return merge(_chunk_dict(r.chunk), Dict("score" => r.score, "rank" => r.rank))
end
