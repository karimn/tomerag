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
