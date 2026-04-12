"""
    query(registry, text; source, content_type=nothing, document_type=nothing,
          top_k=5, score_threshold=0.0f0, embed_backend) -> Vector{QueryResult}

Embed `text` with `embed_backend` and run semantic search against `source`.
Filters and score threshold are applied SQL-side / post-retrieval respectively.
"""
function query(registry::SourceRegistry, text::AbstractString;
               source::AbstractString,
               content_type::Union{Symbol,Nothing}=nothing,
               document_type::Union{Symbol,Nothing}=nothing,
               top_k::Int=5,
               score_threshold::Float32=0.0f0,
               embed_backend::EmbeddingBackend)
    src = get_source(registry, source)
    q = embed(embed_backend, String(text))
    length(q) == src.embedding_dim ||
        error("embedding dim mismatch: source expects $(src.embedding_dim), got $(length(q))")

    filters = Dict{String,Any}()
    content_type === nothing || (filters["content_type"] = String(content_type))
    document_type === nothing || (filters["document_type"] = String(document_type))

    raw = similarity_search(src, q; top_k=top_k, filters=filters)
    score_threshold > 0 || return raw
    return [r for r in raw if r.score >= score_threshold]
end
