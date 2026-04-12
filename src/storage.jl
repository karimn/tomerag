using DBInterface
using DuckDB
using JSON3

"""
    initialize_store(src)

Open `src.db_path`, install+load VSS extension, enable HNSW persistence,
create `source_meta` and `chunks` tables if absent, create HNSW index.
"""
function initialize_store(src::Source)
    db = DBInterface.connect(DuckDB.DB, src.db_path)
    try
        DuckDB.execute(db, "INSTALL vss;")
        DuckDB.execute(db, "LOAD vss;")
        DuckDB.execute(db, "SET hnsw_enable_experimental_persistence = true;")

        DuckDB.execute(db, """
            CREATE TABLE IF NOT EXISTS source_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );
        """)
        DuckDB.execute(db, """
            INSERT INTO source_meta VALUES
                ('source_id',       '$(src.id)'),
                ('embedding_model', '$(src.embedding_model)'),
                ('embedding_dim',   '$(src.embedding_dim)'),
                ('system',          '$(src.system)')
            ON CONFLICT (key) DO UPDATE SET value = excluded.value;
        """)
        DuckDB.execute(db, """
            CREATE TABLE IF NOT EXISTS chunks (
                id               TEXT PRIMARY KEY,
                source_id        TEXT,
                doc_id           TEXT,
                doc_path         TEXT,
                text             TEXT,
                embedding        FLOAT[$(src.embedding_dim)],
                embedding_model  TEXT,
                token_count      INTEGER,
                content_hash     TEXT,
                document_type    TEXT,
                system           TEXT,
                edition          TEXT,
                page             TEXT,
                heading_path     TEXT,
                chunk_order      INTEGER,
                parent_id        TEXT,
                content_type     TEXT,
                tags             TEXT,
                move_trigger     TEXT,
                scene_type       TEXT,
                encounter_key    TEXT,
                npc_name         TEXT,
                license          TEXT
            );
        """)
        DuckDB.execute(db, """
            CREATE UNIQUE INDEX IF NOT EXISTS chunks_dedup_idx
                ON chunks(doc_id, content_hash);
        """)
        DuckDB.execute(db, """
            CREATE INDEX IF NOT EXISTS chunks_hnsw_idx
                ON chunks USING HNSW (embedding) WITH (metric='cosine');
        """)
    finally
        DBInterface.close!(db)
    end
    return nothing
end

"""
    insert_chunks(src, chunks) -> Int

Insert chunks into `src.db_path`. Skips chunks whose `(doc_id, content_hash)`
already exists. Returns the number of rows actually inserted.
"""
function insert_chunks(src::Source, chunks::AbstractVector{Chunk})
    isempty(chunks) && return 0
    db = DBInterface.connect(DuckDB.DB, src.db_path)
    inserted = 0
    try
        DuckDB.execute(db, "LOAD vss;")
        DuckDB.execute(db, "BEGIN;")
        stmt = """
            INSERT INTO chunks VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        for c in chunks
            # Pre-check dedup (DuckDB.jl doesn't expose row-affected count).
            exists = false
            for _ in DuckDB.execute(db,
                    "SELECT 1 FROM chunks WHERE doc_id=? AND content_hash=? LIMIT 1",
                    (c.doc_id, c.content_hash))
                exists = true
            end
            exists && continue
            DuckDB.execute(db, stmt, (
                c.id, c.source_id, c.doc_id, c.doc_path,
                c.text, Vector{Float32}(c.embedding), c.embedding_model, c.token_count, c.content_hash,
                String(c.document_type), c.system, c.edition, c.page,
                JSON3.write(c.heading_path), c.chunk_order, c.parent_id,
                String(c.content_type), JSON3.write(c.tags),
                c.move_trigger,
                c.scene_type === nothing ? nothing : String(c.scene_type),
                c.encounter_key, c.npc_name, String(c.license),
            ))
            inserted += 1
        end
        DuckDB.execute(db, "COMMIT;")
    catch e
        DuckDB.execute(db, "ROLLBACK;")
        rethrow(e)
    finally
        DBInterface.close!(db)
    end
    return inserted
end

"""
    source_stats(src) -> NamedTuple

Returns `(chunk_count::Int, embedding_model::String, embedding_dim::Int)`.
"""
function source_stats(src::Source)
    db = DBInterface.connect(DuckDB.DB, src.db_path)
    try
        cnt = 0
        for row in DuckDB.execute(db, "SELECT COUNT(*) AS n FROM chunks")
            cnt = row.n
        end
        return (chunk_count = Int(cnt),
                embedding_model = src.embedding_model,
                embedding_dim = src.embedding_dim)
    finally
        DBInterface.close!(db)
    end
end

"""
    similarity_search(src, query_embedding; top_k, filters) -> Vector{QueryResult}

Cosine HNSW search. `filters` maps column name to value; supported keys:
`"content_type"`, `"document_type"`, `"system"`, `"doc_id"`.
"""
function similarity_search(src::Source, q::AbstractVector{<:AbstractFloat};
                           top_k::Int=10,
                           filters::Dict{String,Any}=Dict{String,Any}())
    db = DBInterface.connect(DuckDB.DB, src.db_path)
    try
        DuckDB.execute(db, "LOAD vss;")

        where_parts = String[]
        filter_vals = Any[]
        for (col, val) in filters
            col in ("content_type", "document_type", "system", "doc_id") ||
                error("unsupported filter column: $col")
            push!(where_parts, "$col = ?")
            push!(filter_vals, String(val))
        end
        where_clause = isempty(where_parts) ? "" : ("WHERE " * join(where_parts, " AND "))

        # LIMIT is inlined (not a bound param) so DuckDB can use the HNSW index.
        # Cast embedding ARRAY→LIST so DuckDB.jl can deserialise it.
        sql = """
            SELECT id, source_id, doc_id, doc_path, text, embedding::FLOAT[] AS embedding,
                   embedding_model, token_count, content_hash, document_type, system,
                   edition, page, heading_path, chunk_order, parent_id, content_type, tags,
                   move_trigger, scene_type, encounter_key, npc_name, license,
                   array_cosine_distance(embedding, ?::FLOAT[$(src.embedding_dim)]) AS dist
            FROM chunks
            $where_clause
            ORDER BY dist ASC
            LIMIT $top_k
        """
        params = Any[Vector{Float32}(q)]
        append!(params, filter_vals)
        results = QueryResult[]
        rank = 0
        for row in DuckDB.execute(db, sql, params)
            rank += 1
            score = Float32(1.0 - row.dist)
            push!(results, QueryResult(_row_to_chunk(row), score, rank))
        end
        return results
    finally
        DBInterface.close!(db)
    end
end

function _row_to_chunk(row)
    return Chunk(
        id              = row.id,
        source_id       = row.source_id,
        doc_id          = row.doc_id,
        doc_path        = row.doc_path,
        text            = row.text,
        embedding       = Float32[x for x in row.embedding if x !== missing],
        embedding_model = row.embedding_model,
        token_count     = Int(row.token_count),
        content_hash    = row.content_hash,
        document_type   = Symbol(row.document_type),
        system          = row.system,
        edition         = row.edition,
        page            = row.page,
        heading_path    = JSON3.read(row.heading_path, Vector{String}),
        chunk_order     = Int(row.chunk_order),
        parent_id       = row.parent_id === missing ? nothing : row.parent_id,
        content_type    = Symbol(row.content_type),
        tags            = JSON3.read(row.tags, Vector{String}),
        move_trigger    = row.move_trigger === missing ? nothing : row.move_trigger,
        scene_type      = row.scene_type === missing ? nothing : Symbol(row.scene_type),
        encounter_key   = row.encounter_key === missing ? nothing : row.encounter_key,
        npc_name        = row.npc_name === missing ? nothing : row.npc_name,
        license         = Symbol(row.license),
    )
end
