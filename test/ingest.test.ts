import { test, expect } from "bun:test";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { mkdtempSync, writeFileSync } from "node:fs";
import { ingest } from "../src/ingest.ts";
import { initializeStore, sourceStats, bm25Search } from "../src/storage.ts";
import { defaultChunkingConfig } from "../src/types.ts";
import { DEFAULT_CONTENT_TYPES } from "../src/content-types.ts";
import { MockEmbeddingBackend } from "../src/backends/embedding.ts";
import { HeuristicBackend } from "../src/backends/classify.ts";
import { MockExtractionBackend } from "../src/backends/extraction.ts";
import type { Source } from "../src/types.ts";

function mkSource(overrides: Partial<Source> = {}): Source {
  const dir = mkdtempSync(join(tmpdir(), "tomerag-ing-"));
  return {
    id: "iron", name: "Ironsworn", system: "PbtA",
    dbPath: join(dir, "t.duckdb"), embeddingModel: "mock", embeddingDim: 8,
    license: "cc_by",
    chunking: defaultChunkingConfig({ minTokens: 1, maxTokens: 200, overlapTokens: 0 }),
    contentTypes: new Set(DEFAULT_CONTENT_TYPES),
    ...overrides,
  };
}

test("ingest markdown end-to-end with mocks, idempotent on re-ingest", async () => {
  const src = mkSource();
  await initializeStore(src);
  const mdPath = join(mkdtempSync(join(tmpdir(), "tomerag-md-")), "doc.md");
  writeFileSync(mdPath, `# Moves

## Delve the Depths

**When you delve the depths**, roll +wits. On a 10+, choose two.

## Secure an Advantage

**When you secure an advantage**, roll +heart.

## Aid Your Ally

**When you aid your ally**, roll +heart. On a hit, they take +1 on their move.
`);
  const n = await ingest({
    source: src, path: mdPath, docId: "ironsworn-core",
    documentType: "core_rules", format: "markdown",
    embedBackend: new MockEmbeddingBackend({ dim: 8 }),
    classifyBackend: new HeuristicBackend(),
  });
  expect(n).toBe(3);
  expect((await sourceStats(src)).chunkCount).toBe(3);

  const n2 = await ingest({
    source: src, path: mdPath, docId: "ironsworn-core",
    documentType: "core_rules", format: "markdown",
    embedBackend: new MockEmbeddingBackend({ dim: 8 }),
    classifyBackend: new HeuristicBackend(),
  });
  expect(n2).toBe(0);
});

test("ingest pdf with MockExtractionBackend assigns pages, strips markers", async () => {
  const src = mkSource({ id: "pdf-ingest", license: "homebrew" });
  await initializeStore(src);
  const pdfPath = join(mkdtempSync(join(tmpdir(), "tomerag-pdf-")), "f.pdf");
  writeFileSync(pdfPath, "fake pdf bytes");
  const extractor = new MockExtractionBackend([
    { pageNum: 1, text: "# Iron Vow\n**When you swear upon iron**, roll +heart. On a 10+, your vow is strong." },
    { pageNum: 2, text: "# Face Danger\n**When you face danger**, roll +edge. On a miss, pay the price." },
  ]);
  const n = await ingest({
    source: src, path: pdfPath, docId: "test-rules", documentType: "core_rules",
    format: "pdf", embedBackend: new MockEmbeddingBackend({ dim: 8 }),
    classifyBackend: new HeuristicBackend(), extractionBackend: extractor,
  });
  expect(n).toBeGreaterThanOrEqual(1);
  const all = await bm25Search(src, "iron vow face danger", { topK: 20 });
  expect(all.length).toBeGreaterThan(0);
  expect(all.some((r) => r.chunk.page === "1")).toBe(true);
  expect(all.some((r) => r.chunk.page === "2")).toBe(true);
  expect(all.every((r) => !/<!-- page \d+ -->/.test(r.chunk.text))).toBe(true);
});

test("format=auto detects pdf by extension", async () => {
  const src = mkSource({ id: "auto-pdf", license: "homebrew" });
  await initializeStore(src);
  const pdfPath = join(mkdtempSync(join(tmpdir(), "tomerag-auto-")), "f.pdf");
  writeFileSync(pdfPath, "placeholder");
  const n = await ingest({
    source: src, path: pdfPath, docId: "auto-doc", documentType: "core_rules",
    embedBackend: new MockEmbeddingBackend({ dim: 8 }),
    classifyBackend: new HeuristicBackend(),
    extractionBackend: new MockExtractionBackend([{ pageNum: 1, text: "# Content\nAuto-detected pdf content here." }]),
  });
  expect(n).toBeGreaterThanOrEqual(1);
});

test("format=pdf without extractionBackend throws", async () => {
  const src = mkSource();
  await expect(ingest({
    source: src, path: "file.pdf", docId: "x", documentType: "core_rules",
    format: "pdf", embedBackend: new MockEmbeddingBackend({ dim: 8 }),
    classifyBackend: new HeuristicBackend(),
  })).rejects.toThrow(/extractionBackend is required/);
});
