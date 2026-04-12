"""
    _rrf_merge(dense, bm25; k, top_k, score_threshold) -> Vector{QueryResult}

Reciprocal Rank Fusion: score(chunk) = Σ 1/(k + rank_i).
`dense` is Vector{QueryResult} from similarity_search.
`bm25` is Vector{Tuple{Chunk,Float32}} from bm25_search (scores ignored, only rank used).
"""
function _rrf_merge(dense::Vector{QueryResult},
                    bm25::Vector{Tuple{Chunk,Float32}};
                    k::Int=60, top_k::Int=5,
                    score_threshold::Float32=0.0f0)
    scores = Dict{String,Float32}()
    chunks = Dict{String,Chunk}()

    for (i, qr) in enumerate(dense)
        id = qr.chunk.id
        scores[id] = get(scores, id, 0.0f0) + 1.0f0 / (k + i)
        chunks[id] = qr.chunk
    end
    for (i, (chunk, _)) in enumerate(bm25)
        id = chunk.id
        scores[id] = get(scores, id, 0.0f0) + 1.0f0 / (k + i)
        chunks[id] = chunk
    end

    sorted_ids = sort(collect(keys(scores)), by = id -> -scores[id])
    out = QueryResult[]
    for id in sorted_ids
        s = scores[id]
        s >= score_threshold || continue
        push!(out, QueryResult(chunks[id], s, length(out) + 1))
        length(out) >= top_k && break
    end
    return out
end

"""
    query(registry, text; source, content_type, document_type, top_k,
          score_threshold, embed_backend, mode) -> Vector{QueryResult}

Embed `text` and search `source`. `mode=:hybrid` (default) merges dense HNSW and
BM25 via Reciprocal Rank Fusion. `mode=:dense` runs dense-only.
"""
function query(registry::SourceRegistry, text::AbstractString;
               source::AbstractString,
               content_type::Union{Symbol,Nothing}=nothing,
               document_type::Union{Symbol,Nothing}=nothing,
               top_k::Int=5,
               score_threshold::Float32=0.0f0,
               embed_backend::EmbeddingBackend,
               mode::Symbol=:hybrid)
    src = get_source(registry, source)
    q = embed(embed_backend, String(text))
    length(q) == src.embedding_dim ||
        error("embedding dim mismatch: source expects $(src.embedding_dim), got $(length(q))")

    filters = Dict{String,Any}()
    content_type === nothing || (filters["content_type"] = String(content_type))
    document_type === nothing || (filters["document_type"] = String(document_type))

    if mode === :dense
        raw = similarity_search(src, q; top_k=top_k, filters=filters)
        score_threshold > 0 || return raw
        return [r for r in raw if r.score >= score_threshold]
    else
        dense = similarity_search(src, q; top_k=top_k * 2, filters=filters)
        bm25  = bm25_search(src, String(text); top_k=top_k * 2, filters=filters)
        return _rrf_merge(dense, bm25; top_k=top_k, score_threshold=score_threshold)
    end
end
