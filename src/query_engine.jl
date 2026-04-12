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

"""
    filter_chunks(registry, source; content_type, document_type, top_k, offset) -> Vector{Chunk}

Pure metadata filter — no embedding, no text query. Paginates via `offset`.
"""
function filter_chunks(registry::SourceRegistry, source::AbstractString;
                       content_type::Union{Symbol,Nothing}=nothing,
                       document_type::Union{Symbol,Nothing}=nothing,
                       top_k::Int=100, offset::Int=0)
    src = get_source(registry, source)
    db = DBInterface.connect(DuckDB.DB, src.db_path)
    try
        where_parts = String[]
        vals = Any[]
        content_type === nothing || (push!(where_parts, "content_type = ?"); push!(vals, String(content_type)))
        document_type === nothing || (push!(where_parts, "document_type = ?"); push!(vals, String(document_type)))
        where = isempty(where_parts) ? "" : "WHERE " * join(where_parts, " AND ")
        sql = """
            SELECT id, source_id, doc_id, doc_path, text, embedding::FLOAT[] AS embedding,
                   embedding_model, token_count, content_hash, document_type, system,
                   edition, page, heading_path, chunk_order, parent_id, content_type, tags,
                   move_trigger, scene_type, encounter_key, npc_name, license
            FROM chunks $where ORDER BY chunk_order ASC
            LIMIT $top_k OFFSET $offset
        """
        result = Chunk[]
        row_iter = isempty(vals) ? DuckDB.execute(db, sql) : DuckDB.execute(db, sql, vals)
        for row in row_iter
            push!(result, _row_to_chunk(row))
        end
        return result
    finally
        DBInterface.close!(db)
    end
end

"""
    lookup(registry, name; source, content_type) -> Vector{QueryResult}

BM25 name lookup against chunk text. Returns results ranked by BM25 score.
"""
function lookup(registry::SourceRegistry, name::AbstractString;
                source::AbstractString,
                content_type::Union{Symbol,Nothing}=nothing)
    src = get_source(registry, source)
    db = DBInterface.connect(DuckDB.DB, src.db_path)
    try
        DuckDB.query(db, "LOAD fts")
        where_parts = ["score IS NOT NULL"]
        vals = Any[]
        content_type === nothing || (push!(where_parts, "content_type = ?"); push!(vals, String(content_type)))
        where = "WHERE " * join(where_parts, " AND ")
        escaped = replace(String(name), "'" => "''")
        sql = """
            SELECT id, source_id, doc_id, doc_path, text, embedding::FLOAT[] AS embedding,
                   embedding_model, token_count, content_hash, document_type, system,
                   edition, page, heading_path, chunk_order, parent_id, content_type, tags,
                   move_trigger, scene_type, encounter_key, npc_name, license,
                   fts_main_chunks.match_bm25(id, '$escaped') AS score
            FROM chunks $where ORDER BY score DESC LIMIT 20
        """
        results = QueryResult[]
        rank = 0
        row_iter = isempty(vals) ? DuckDB.execute(db, sql) : DuckDB.execute(db, sql, vals)
        for row in row_iter
            rank += 1
            push!(results, QueryResult(_row_to_chunk(row), Float32(row.score), rank))
        end
        return results
    finally
        DBInterface.close!(db)
    end
end

"""
    multi_query(registry, queries; embed_backend, top_k) -> Vector{QueryResult}

Runs multiple `query()` calls, min-max normalizes scores per sub-query,
merges and deduplicates by chunk ID (highest score wins), returns top_k.

`queries` is a `Vector` of `(text::String, kwargs::NamedTuple)` pairs.
`embed_backend` is shared across all sub-queries.
"""
function multi_query(registry::SourceRegistry,
                     queries::Vector{<:Tuple{<:AbstractString,<:NamedTuple}};
                     embed_backend::EmbeddingBackend,
                     top_k::Int=10)
    all_results = QueryResult[]
    for (text, kwargs) in queries
        results = query(registry, text; embed_backend=embed_backend,
                        top_k=top_k * 2, kwargs...)
        isempty(results) && continue
        scores = Float32[r.score for r in results]
        lo, hi = minimum(scores), maximum(scores)
        normed = hi > lo ? (scores .- lo) ./ (hi - lo) : ones(Float32, length(scores))
        for (r, s) in zip(results, normed)
            push!(all_results, QueryResult(r.chunk, s, 0))
        end
    end
    best = Dict{String,QueryResult}()
    for qr in all_results
        if !haskey(best, qr.chunk.id) || qr.score > best[qr.chunk.id].score
            best[qr.chunk.id] = qr
        end
    end
    sorted = sort(collect(values(best)), by = r -> -r.score)
    return [QueryResult(r.chunk, r.score, i) for (i, r) in enumerate(first(sorted, top_k))]
end

"""
    get_context(registry, chunk_id; source, before, after) -> Vector{Chunk}

Fetches `chunk_id` plus `before` chunks before and `after` chunks after it,
ordered by `chunk_order` within the same document.
"""
function get_context(registry::SourceRegistry, chunk_id::AbstractString;
                     source::AbstractString,
                     before::Int=1, after::Int=1)
    src = get_source(registry, source)
    db = DBInterface.connect(DuckDB.DB, src.db_path)
    try
        doc_id = nothing
        order  = nothing
        for row in DuckDB.execute(db,
                "SELECT doc_id, chunk_order FROM chunks WHERE id = ? LIMIT 1",
                (String(chunk_id),))
            doc_id = row.doc_id
            order  = Int(row.chunk_order)
        end
        isnothing(doc_id) && return Chunk[]

        sql = """
            SELECT id, source_id, doc_id, doc_path, text, embedding::FLOAT[] AS embedding,
                   embedding_model, token_count, content_hash, document_type, system,
                   edition, page, heading_path, chunk_order, parent_id, content_type, tags,
                   move_trigger, scene_type, encounter_key, npc_name, license
            FROM chunks
            WHERE doc_id = ? AND chunk_order >= ? AND chunk_order <= ?
            ORDER BY chunk_order ASC
        """
        result = Chunk[]
        for row in DuckDB.execute(db, sql, (doc_id, order - before, order + after))
            push!(result, _row_to_chunk(row))
        end
        return result
    finally
        DBInterface.close!(db)
    end
end
