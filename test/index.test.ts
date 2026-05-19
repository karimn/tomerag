import { describe, it, expect } from "bun:test";
import {
  // types.ts
  defaultChunkingConfig,
  // content-types.ts
  DEFAULT_CONTENT_TYPES,
  // tokenize.ts
  wsTokens,
  tokenCount,
  contentHash,
  // chunker.ts
  chunkDocument,
  // backends/embedding.ts
  MockEmbeddingBackend,
  // backends/classify.ts
  MockClassifyBackend,
  // backends/extraction.ts
  MockExtractionBackend,
  // storage.ts
  initializeStore,
  // ingest.ts
  ingest,
} from "../src/index.ts";

describe("index barrel re-exports", () => {
  it("defaultChunkingConfig is a function", () => {
    expect(typeof defaultChunkingConfig).toBe("function");
  });

  it("DEFAULT_CONTENT_TYPES is a Set", () => {
    expect(DEFAULT_CONTENT_TYPES).toBeInstanceOf(Set);
  });

  it("wsTokens is a function", () => {
    expect(typeof wsTokens).toBe("function");
  });

  it("tokenCount is a function", () => {
    expect(typeof tokenCount).toBe("function");
  });

  it("contentHash is a function", () => {
    expect(typeof contentHash).toBe("function");
  });

  it("chunkDocument is a function", () => {
    expect(typeof chunkDocument).toBe("function");
  });

  it("MockEmbeddingBackend is a class (constructor)", () => {
    expect(typeof MockEmbeddingBackend).toBe("function");
    expect(new MockEmbeddingBackend()).toBeInstanceOf(MockEmbeddingBackend);
  });

  it("MockClassifyBackend is a class", () => {
    expect(typeof MockClassifyBackend).toBe("function");
    expect(new MockClassifyBackend()).toBeInstanceOf(MockClassifyBackend);
  });

  it("MockExtractionBackend is a class", () => {
    expect(typeof MockExtractionBackend).toBe("function");
    expect(new MockExtractionBackend([])).toBeInstanceOf(MockExtractionBackend);
  });

  it("initializeStore is a function", () => {
    expect(typeof initializeStore).toBe("function");
  });

  it("ingest is a function", () => {
    expect(typeof ingest).toBe("function");
  });
});

// Type-only imports compile check (no runtime assertions needed)
import type {
  License,
  DocumentType,
  ContentType,
  Overflow,
  ChunkingConfig,
  Chunk,
  QueryResult,
  Source,
  RawChunk,
  PageText,
  Section,
  Classification,
  IngestOptions,
  ClaudeBackendOptions,
  CostEstimate,
} from "../src/index.ts";

// Use types to prevent "unused" errors — assigned to typed variables
const _license: License = "cc_by";
const _docType: DocumentType = "core_rules";
const _contentType: ContentType = "mechanic";
const _overflow: Overflow = "paragraph";
const _config: ChunkingConfig = defaultChunkingConfig();
const _chunk: Chunk | undefined = undefined;
const _queryResult: QueryResult | undefined = undefined;
const _source: Source | undefined = undefined;
const _rawChunk: RawChunk | undefined = undefined;
const _pageText: PageText | undefined = undefined;
const _section: Section | undefined = undefined;
const _classification: Classification | undefined = undefined;
const _ingestOptions: IngestOptions | undefined = undefined;
const _claudeBackendOptions: ClaudeBackendOptions | undefined = undefined;
const _costEstimate: CostEstimate | undefined = undefined;

// Suppress unused variable warnings
void [
  _license, _docType, _contentType, _overflow, _config, _chunk, _queryResult,
  _source, _rawChunk, _pageText, _section, _classification, _ingestOptions,
  _claudeBackendOptions, _costEstimate,
];
