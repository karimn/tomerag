import { test, expect } from "bun:test";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { mkdtempSync } from "node:fs";
import { DuckDBInstance } from "@duckdb/node-api";
import { ingest } from "../src/ingest.ts";
import { initializeStore, sourceStats } from "../src/storage.ts";
import { defaultChunkingConfig } from "../src/types.ts";
import { PBTA_CONTENT_TYPES } from "../src/content-types.ts";
import { MockEmbeddingBackend } from "../src/backends/embedding.ts";
import { HeuristicBackend } from "../src/backends/classify.ts";
import type { Source } from "../src/types.ts";

const FIXTURE = join(import.meta.dir, "fixtures", "ironsworn_sample.md");

function mkSource(dim: number): Source {
  return {
    id: "iron", name: "Ironsworn", system: "PbtA",
    dbPath: join(mkdtempSync(join(tmpdir(), "tomerag-int-")), "ironsworn.duckdb"),
    embeddingModel: "mock", embeddingDim: dim, license: "cc_by",
    chunking: defaultChunkingConfig({ minTokens: 5, maxTokens: 300 }),
    contentTypes: new Set(PBTA_CONTENT_TYPES),
  };
}

test("ingest produces a schema the scribe reader's SQL can consume", async () => {
  const dim = 16;
  const src = mkSource(dim);
  await initializeStore(src);
  const n = await ingest({
    source: src, path: FIXTURE, docId: "ironsworn-sample",
    documentType: "core_rules", format: "markdown",
    embedBackend: new MockEmbeddingBackend({ dim }),
    classifyBackend: new HeuristicBackend(),
  });
  expect(n).toBeGreaterThanOrEqual(4);
  expect((await sourceStats(src)).chunkCount).toBe(n);

  const instance = await DuckDBInstance.create(src.dbPath);
  const conn = await instance.connect();
  try {
    const q = await new MockEmbeddingBackend({ dim }).embed("delve the depths");
    const embLit = `[${Array.from(q).join(",")}]::FLOAT[${dim}]`;
    const vec = await conn.runAndReadAll(
      `SELECT id, text, heading_path, content_type, move_trigger, page,
              array_cosine_similarity(embedding, ${embLit}) AS score
       FROM chunks ORDER BY score DESC LIMIT 3`,
    );
    const vrows = vec.getRowObjectsJS() as Record<string, unknown>[];
    expect(vrows.length).toBeGreaterThanOrEqual(1);
    expect(String(vrows[0]!["text"]).length).toBeGreaterThan(0);
    expect(typeof vrows[0]!["heading_path"]).toBe("string"); // JSON-encoded
    expect(JSON.parse(String(vrows[0]!["heading_path"]))).toBeInstanceOf(Array);

    const bm = await conn.runAndReadAll(
      `SELECT id, content_type, move_trigger,
              fts_main_chunks.match_bm25(id, 'delve the depths') AS score
       FROM chunks
       WHERE fts_main_chunks.match_bm25(id, 'delve the depths') IS NOT NULL
       ORDER BY score DESC LIMIT 5`,
    );
    expect((bm.getRowObjectsJS() as unknown[]).length).toBeGreaterThanOrEqual(1);

    const moves = await conn.runAndReadAll(
      `SELECT content_type, move_trigger FROM chunks WHERE content_type = 'move'`,
    );
    const mrows = moves.getRowObjectsJS() as Record<string, unknown>[];
    expect(mrows.length).toBeGreaterThanOrEqual(2);
    expect(mrows.every((r) => r["content_type"] === "move")).toBe(true);
  } finally {
    conn.closeSync();
  }
});

test.skipIf(process.env["TOMERAG_LIVE_TESTS"] !== "1")(
  "literal round-trip through scribe/src/rag/query.ts (Ollama + 768-dim)",
  async () => {
    const SCRIBE_QUERY = "/media/karim/Code-Drive/karimn-code/rpg-rules/plugins/ironsworn/scribe/src/rag/query.ts";
    const dim = 768;
    const src = mkSource(dim);
    await initializeStore(src);
    await ingest({
      source: src, path: FIXTURE, docId: "ironsworn-sample",
      documentType: "core_rules", format: "markdown",
      embedBackend: new MockEmbeddingBackend({ dim }),
      classifyBackend: new HeuristicBackend(),
    });
    process.env["DB_PATH"] = src.dbPath;
    const { searchRules } = (await import(SCRIBE_QUERY)) as any;
    const results = await searchRules("delve the depths", { k: 3 });
    expect(results.length).toBeGreaterThanOrEqual(1);
    expect(results[0]!).toHaveProperty("headingPath");
    expect(results[0]!).toHaveProperty("moveTrigger");
  },
);
