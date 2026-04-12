using Oxygen
using HTTP
using JSON3
using DBInterface

"""
    serve(registry; port=8080, async=false, embed_backend=nothing)

Start the TomeRAG REST API server.
- `async=true` returns a closeable handle (use `close(server)` to stop).
- `embed_backend` is required for /query, /lookup, /multi_query endpoints.
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
