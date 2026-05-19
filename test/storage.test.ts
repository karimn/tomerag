import { test, expect } from "bun:test";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { mkdtempSync } from "node:fs";
import {
  initializeStore, insertChunks, sourceStats, similaritySearch, bm25Search,
} from "../src/storage.ts";
import { defaultChunkingConfig } from "../src/types.ts";
import { DEFAULT_CONTENT_TYPES } from "../src/content-types.ts";
import type { Chunk, Source } from "../src/types.ts";

function tmpDb(): string {
  return join(mkdtempSync(join(tmpdir(), "tomerag-db-")), "t.duckdb");
}

function mkSource(path: string, dim = 4): Source {
  return {
    id: "coriolis", name: "Coriolis", system: "YZE", dbPath: path,
    embeddingModel: "mock", embeddingDim: dim, license: "homebrew",
    chunking: defaultChunkingConfig(), contentTypes: new Set(DEFAULT_CONTENT_TYPES),
  };
}

function mkChunk(id: string, text: string, dim = 4): Chunk {
  return {
    id, sourceId: "coriolis", docId: "doc", docPath: "/tmp/doc.md", text,
    embedding: Float32Array.from([0.1, 0.2, 0.3, 0.4].slice(0, dim)),
    embeddingModel: "mock", tokenCount: text.split(/\s+/).length, contentHash: `h_${id}`,
    documentType: "core_rules", system: "YZE", edition: "1e", page: "",
    headingPath: ["Ch"], chunkOrder: 0, parentId: null, contentType: "mechanic",
    tags: [], moveTrigger: null, sceneType: null, encounterKey: null,
    npcName: null, license: "homebrew",
  };
}

test("initialize + insert + stats", async () => {
  const src = mkSource(tmpDb());
  await initializeStore(src);
  const n = await insertChunks(src, [mkChunk("a", "hello world"), mkChunk("b", "goodbye world")]);
  expect(n).toBe(2);
  const s = await sourceStats(src);
  expect(s.chunkCount).toBe(2);
  expect(s.embeddingModel).toBe("mock");
  expect(s.embeddingDim).toBe(4);
});

test("dedup scoped to (docId, contentHash)", async () => {
  const src = mkSource(tmpDb());
  await initializeStore(src);
  const c = mkChunk("a", "hello world");
  expect(await insertChunks(src, [c])).toBe(1);
  expect(await insertChunks(src, [c])).toBe(0);
  expect((await sourceStats(src)).chunkCount).toBe(1);
});

test("similaritySearch ranks by cosine", async () => {
  const src = mkSource(tmpDb());
  await initializeStore(src);
  const c1 = { ...mkChunk("a", "blight corruption spreads"), embedding: Float32Array.from([1, 0, 0, 0]) };
  const c2 = { ...mkChunk("b", "delve the deeps"), embedding: Float32Array.from([0, 1, 0, 0]) };
  await insertChunks(src, [c1, c2]);
  const results = await similaritySearch(src, Float32Array.from([0.9, 0.1, 0, 0]), { topK: 2 });
  expect(results.length).toBe(2);
  expect(results[0]!.chunk.id).toBe("a");
  expect(results[0]!.rank).toBe(1);
  expect(results[0]!.score).toBeGreaterThanOrEqual(0);
  expect(results[0]!.score).toBeLessThanOrEqual(1);
});

test("bm25Search ranks and filters", async () => {
  const src = mkSource(tmpDb());
  await initializeStore(src);
  await insertChunks(src, [
    mkChunk("bm1", "iron vow momentum move roll"),
    mkChunk("bm2", "delve the dungeon depths explore"),
    mkChunk("bm3", "blight corruption spreads darkness"),
  ]);
  const results = await bm25Search(src, "momentum iron vow", { topK: 3 });
  expect(results.length).toBeGreaterThanOrEqual(1);
  expect(results[0]!.chunk.id).toBe("bm1");
  expect(results[0]!.score).toBeGreaterThan(0);

  const c4 = { ...mkChunk("bm4", "iron vow move lore"), contentType: "lore" as const };
  await insertChunks(src, [c4]);
  const lore = await bm25Search(src, "iron vow", { topK: 5, filters: { content_type: "lore" } });
  expect(lore.every((r) => r.chunk.contentType === "lore")).toBe(true);
});
