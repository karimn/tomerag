import { test, expect } from "bun:test";
import type { Chunk, QueryResult, ChunkingConfig, Source } from "../src/types.ts";
import { defaultChunkingConfig } from "../src/types.ts";
import { DEFAULT_CONTENT_TYPES } from "../src/content-types.ts";

test("defaultChunkingConfig has Julia defaults", () => {
  const cfg: ChunkingConfig = defaultChunkingConfig();
  expect(cfg.minTokens).toBe(100);
  expect(cfg.maxTokens).toBe(800);
  expect(cfg.overlapTokens).toBe(50);
  expect(cfg.overflow).toBe("paragraph");
  expect(cfg.atomicPatterns).toEqual([]);
});

test("Chunk and QueryResult are structurally usable", () => {
  const c: Chunk = {
    id: "abc", sourceId: "coriolis", docId: "core-1e", docPath: "/tmp/x.md",
    text: "hello", embedding: new Float32Array([0.1, 0.2]), embeddingModel: "mock",
    tokenCount: 1, contentHash: "h", documentType: "core_rules", system: "YZE",
    edition: "1e", page: "1", headingPath: ["Ch", "Sec"], chunkOrder: 0,
    parentId: null, contentType: "mechanic", tags: ["x"], moveTrigger: null,
    sceneType: null, encounterKey: null, npcName: null, license: "homebrew",
  };
  expect(c.id).toBe("abc");
  expect(c.parentId).toBeNull();
  expect(c.embedding.length).toBe(2);

  const r: QueryResult = { chunk: c, score: 0.9, rank: 1 };
  expect(r.score).toBeCloseTo(0.9);
  expect(r.rank).toBe(1);
});

test("Source carries a content-type set", () => {
  const s: Source = {
    id: "coriolis", name: "Coriolis: The Great Dark", system: "YZE",
    dbPath: "/tmp/x.duckdb", embeddingModel: "mock", embeddingDim: 4,
    license: "homebrew", chunking: defaultChunkingConfig(),
    contentTypes: new Set(DEFAULT_CONTENT_TYPES),
  };
  expect(s.contentTypes.has("mechanic")).toBe(true);
});
