import { DuckDBInstance } from "@duckdb/node-api";
import type { DuckDBConnection } from "@duckdb/node-api";
import type { Chunk, ContentType, DocumentType, License, QueryResult, ReadSource } from "./types.ts";

type FilterCol = "content_type" | "document_type" | "system" | "doc_id";
const ALLOWED_FILTERS: ReadonlySet<string> = new Set(["content_type", "document_type", "system", "doc_id"]);

/**
 * Opens a DuckDB connection, runs `fn`, then closes both the connection and the
 * underlying instance — preventing the file-handle / memory leak that occurs
 * when only the connection is closed.
 */
export async function withConnection<T>(
  dbPath: string,
  fn: (conn: DuckDBConnection) => Promise<T>,
): Promise<T> {
  const instance = await DuckDBInstance.create(dbPath);
  const conn = await instance.connect();
  try {
    return await fn(conn);
  } finally {
    conn.closeSync();
    instance.closeSync();
  }
}

function embeddingLiteral(emb: Float32Array, dim: number): string {
  return `[${Array.from(emb).join(",")}]::FLOAT[${dim}]`;
}

export async function initializeStore(src: ReadSource): Promise<void> {
  await withConnection(src.dbPath, async (conn) => {
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
  });
}

export async function sourceStats(
  src: ReadSource,
): Promise<{ chunkCount: number; embeddingModel: string; embeddingDim: number }> {
  return withConnection(src.dbPath, async (conn) => {
    const r = await conn.runAndReadAll("SELECT COUNT(*) AS n FROM chunks");
    const row = r.getRowObjectsJS()[0]!;
    return {
      chunkCount: Number(row["n"]),
      embeddingModel: src.embeddingModel,
      embeddingDim: src.embeddingDim,
    };
  });
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
  src: ReadSource,
  queryEmbedding: Float32Array,
  opts?: { topK?: number; filters?: Partial<Record<FilterCol, string>> },
): Promise<QueryResult[]> {
  const topK = opts?.topK ?? 10;
  const { clause, values } = buildFilters(opts?.filters);
  return withConnection(src.dbPath, async (conn) => {
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
  });
}

export async function bm25Search(
  src: ReadSource,
  queryText: string,
  opts?: { topK?: number; filters?: Partial<Record<FilterCol, string>> },
): Promise<Array<{ chunk: Chunk; score: number }>> {
  const topK = opts?.topK ?? 10;
  const { clause, values } = buildFilters(opts?.filters);
  return withConnection(src.dbPath, async (conn) => {
    await conn.run("LOAD fts;");
    const where = clause === ""
      ? "WHERE score IS NOT NULL"
      : clause.replace("WHERE ", "WHERE score IS NOT NULL AND ");
    const sql = `
      SELECT ${ROW_COLUMNS},
        fts_main_chunks.match_bm25(id, ?) AS score
      FROM chunks
      ${where}
      ORDER BY score DESC
      LIMIT ${topK}`;
    const r = await conn.runAndReadAll(sql, [queryText, ...values]);
    const rows = r.getRowObjectsJS() as Record<string, unknown>[];
    return rows.map((row) => ({ chunk: rowToChunk(row), score: Number(row["score"]) }));
  });
}
