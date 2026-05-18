import { DuckDBInstance } from "@duckdb/node-api";
import type { DuckDBConnection } from "@duckdb/node-api";
import type { Chunk, ContentType, DocumentType, License, QueryResult, Source } from "./types.ts";

type FilterCol = "content_type" | "document_type" | "system" | "doc_id";
const ALLOWED_FILTERS: ReadonlySet<string> = new Set(["content_type", "document_type", "system", "doc_id"]);

async function connect(dbPath: string): Promise<DuckDBConnection> {
  const instance = await DuckDBInstance.create(dbPath);
  const conn = await instance.connect();
  return conn;
}

function embeddingLiteral(emb: Float32Array, dim: number): string {
  return `[${Array.from(emb).join(",")}]::FLOAT[${dim}]`;
}

export async function initializeStore(src: Source): Promise<void> {
  const conn = await connect(src.dbPath);
  try {
    await conn.run("INSTALL vss;");
    await conn.run("LOAD vss;");
    await conn.run("SET hnsw_enable_experimental_persistence = true;");
    await conn.run("CREATE TABLE IF NOT EXISTS source_meta (key TEXT PRIMARY KEY, value TEXT);");
    await conn.run(
      `INSERT INTO source_meta VALUES ('source_id', ?), ('embedding_model', ?),
       ('embedding_dim', ?), ('system', ?)
       ON CONFLICT (key) DO UPDATE SET value = excluded.value;`,
      [src.id, src.embeddingModel, String(src.embeddingDim), src.system],
    );
    await conn.run(`
      CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY, source_id TEXT, doc_id TEXT, doc_path TEXT, text TEXT,
        embedding FLOAT[${src.embeddingDim}], embedding_model TEXT, token_count INTEGER,
        content_hash TEXT, document_type TEXT, system TEXT, edition TEXT, page TEXT,
        heading_path TEXT, chunk_order INTEGER, parent_id TEXT, content_type TEXT,
        tags TEXT, move_trigger TEXT, scene_type TEXT, encounter_key TEXT,
        npc_name TEXT, license TEXT
      );`);
    await conn.run("CREATE UNIQUE INDEX IF NOT EXISTS chunks_dedup_idx ON chunks(doc_id, content_hash);");
    await conn.run("CREATE INDEX IF NOT EXISTS chunks_hnsw_idx ON chunks USING HNSW (embedding) WITH (metric='cosine');");
    await conn.run("INSTALL fts;");
    await conn.run("LOAD fts;");
    await conn.run("PRAGMA create_fts_index('chunks', 'id', 'text', stemmer='porter', overwrite=1);");
  } finally {
    conn.closeSync();
  }
}

export async function insertChunks(src: Source, chunks: Chunk[]): Promise<number> {
  if (chunks.length === 0) return 0;
  const conn = await connect(src.dbPath);
  let inserted = 0;
  try {
    await conn.run("LOAD vss;");
    await conn.run("LOAD fts;");
    await conn.run("BEGIN;");
    for (const c of chunks) {
      const existing = await conn.runAndReadAll(
        "SELECT 1 FROM chunks WHERE doc_id = ? AND content_hash = ? LIMIT 1",
        [c.docId, c.contentHash],
      );
      if (existing.getRowObjectsJS().length > 0) continue;
      await conn.run(
        `INSERT INTO chunks VALUES
         (?, ?, ?, ?, ?, ${embeddingLiteral(c.embedding, src.embeddingDim)},
          ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        [
          c.id, c.sourceId, c.docId, c.docPath, c.text,
          c.embeddingModel, c.tokenCount, c.contentHash, c.documentType,
          c.system, c.edition, c.page, JSON.stringify(c.headingPath),
          c.chunkOrder, c.parentId, c.contentType, JSON.stringify(c.tags),
          c.moveTrigger, c.sceneType, c.encounterKey, c.npcName, c.license,
        ],
      );
      inserted++;
    }
    await conn.run("COMMIT;");
    if (inserted > 0) {
      await conn.run("PRAGMA create_fts_index('chunks', 'id', 'text', stemmer='porter', overwrite=1);");
    }
  } catch (e) {
    try { await conn.run("ROLLBACK;"); } catch { /* ignore */ }
    throw e;
  } finally {
    conn.closeSync();
  }
  return inserted;
}

export async function sourceStats(
  src: Source,
): Promise<{ chunkCount: number; embeddingModel: string; embeddingDim: number }> {
  const conn = await connect(src.dbPath);
  try {
    const r = await conn.runAndReadAll("SELECT COUNT(*) AS n FROM chunks");
    const row = r.getRowObjectsJS()[0]!;
    return {
      chunkCount: Number(row["n"]),
      embeddingModel: src.embeddingModel,
      embeddingDim: src.embeddingDim,
    };
  } finally {
    conn.closeSync();
  }
}

const ROW_COLUMNS = `id, source_id, doc_id, doc_path, text, embedding::FLOAT[] AS embedding,
  embedding_model, token_count, content_hash, document_type, system, edition, page,
  heading_path, chunk_order, parent_id, content_type, tags, move_trigger, scene_type,
  encounter_key, npc_name, license`;

function jsonArray(value: unknown): string[] {
  if (typeof value !== "string") return [];
  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed) ? parsed.map((x) => String(x)) : [];
  } catch {
    return [];
  }
}

export function rowToChunk(row: Record<string, unknown>): Chunk {
  const emb = row["embedding"];
  const arr = Array.isArray(emb) ? emb.filter((x) => x != null).map((x) => Number(x)) : [];
  const str = (v: unknown): string | null => (v == null ? null : String(v));
  return {
    id: String(row["id"]),
    sourceId: String(row["source_id"]),
    docId: String(row["doc_id"]),
    docPath: String(row["doc_path"]),
    text: String(row["text"]),
    embedding: Float32Array.from(arr),
    embeddingModel: String(row["embedding_model"]),
    tokenCount: Number(row["token_count"]),
    contentHash: String(row["content_hash"]),
    documentType: String(row["document_type"]) as DocumentType,
    system: String(row["system"]),
    edition: String(row["edition"]),
    page: row["page"] == null ? "" : String(row["page"]),
    headingPath: jsonArray(row["heading_path"]),
    chunkOrder: Number(row["chunk_order"]),
    parentId: str(row["parent_id"]),
    contentType: String(row["content_type"]) as ContentType,
    tags: jsonArray(row["tags"]),
    moveTrigger: str(row["move_trigger"]),
    sceneType: str(row["scene_type"]),
    encounterKey: str(row["encounter_key"]),
    npcName: str(row["npc_name"]),
    license: String(row["license"]) as License,
  };
}

function buildFilters(filters?: Partial<Record<FilterCol, string>>): { clause: string; values: string[] } {
  if (!filters) return { clause: "", values: [] };
  const parts: string[] = [];
  const values: string[] = [];
  for (const [col, val] of Object.entries(filters)) {
    if (!ALLOWED_FILTERS.has(col)) throw new Error(`unsupported filter column: ${col}`);
    if (val === undefined) continue;
    parts.push(`${col} = ?`);
    values.push(String(val));
  }
  return { clause: parts.length === 0 ? "" : "WHERE " + parts.join(" AND "), values };
}

export async function similaritySearch(
  src: Source,
  queryEmbedding: Float32Array,
  opts?: { topK?: number; filters?: Partial<Record<FilterCol, string>> },
): Promise<QueryResult[]> {
  const topK = opts?.topK ?? 10;
  const { clause, values } = buildFilters(opts?.filters);
  const conn = await connect(src.dbPath);
  try {
    await conn.run("LOAD vss;");
    const sql = `
      SELECT ${ROW_COLUMNS},
        array_cosine_distance(embedding, ${embeddingLiteral(queryEmbedding, src.embeddingDim)}) AS dist
      FROM chunks
      ${clause}
      ORDER BY dist ASC
      LIMIT ${topK}`;
    const r = await conn.runAndReadAll(sql, values);
    const rows = r.getRowObjectsJS() as Record<string, unknown>[];
    return rows.map((row, i) => ({
      chunk: rowToChunk(row),
      score: 1 - Number(row["dist"]),
      rank: i + 1,
    }));
  } finally {
    conn.closeSync();
  }
}

export async function bm25Search(
  src: Source,
  queryText: string,
  opts?: { topK?: number; filters?: Partial<Record<FilterCol, string>> },
): Promise<Array<{ chunk: Chunk; score: number }>> {
  const topK = opts?.topK ?? 10;
  const { clause, values } = buildFilters(opts?.filters);
  const conn = await connect(src.dbPath);
  try {
    await conn.run("LOAD fts;");
    const escaped = queryText.replace(/'/g, "''");
    const where = clause === ""
      ? "WHERE score IS NOT NULL"
      : clause.replace("WHERE ", "WHERE score IS NOT NULL AND ");
    const sql = `
      SELECT ${ROW_COLUMNS},
        fts_main_chunks.match_bm25(id, '${escaped}') AS score
      FROM chunks
      ${where}
      ORDER BY score DESC
      LIMIT ${topK}`;
    const r = await conn.runAndReadAll(sql, values);
    const rows = r.getRowObjectsJS() as Record<string, unknown>[];
    return rows.map((row) => ({ chunk: rowToChunk(row), score: Number(row["score"]) }));
  } finally {
    conn.closeSync();
  }
}
