// Public barrel re-exports for tomerag-ts

// ── types ────────────────────────────────────────────────────────────────────
export type {
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
} from "./types.ts";
export { defaultChunkingConfig } from "./types.ts";

// ── content-types ─────────────────────────────────────────────────────────────
export { DEFAULT_CONTENT_TYPES, PBTA_CONTENT_TYPES, YZE_CONTENT_TYPES } from "./content-types.ts";

// ── tokenize ──────────────────────────────────────────────────────────────────
// NOTE: wsCount is intentionally NOT exported (it is a private internal function)
export { wsTokens, normalizeText, tokenCount, contentHash } from "./tokenize.ts";

// ── chunker ───────────────────────────────────────────────────────────────────
export { parseMarkdownSections, splitToTokenBudget, chunkDocument } from "./chunker.ts";

// ── config ────────────────────────────────────────────────────────────────────
export { loadConfig, resetConfigCache, requireAnthropicKey } from "./config.ts";

// ── backends/embedding ────────────────────────────────────────────────────────
export { EmbeddingBackend, MockEmbeddingBackend, OllamaBackend } from "./backends/embedding.ts";

// ── backends/classify ─────────────────────────────────────────────────────────
export type { ClaudeBackendOptions, CostEstimate } from "./backends/classify.ts";
export {
  ClassifyBackend,
  MockClassifyBackend,
  HeuristicBackend,
  ClaudeBackend,
  forecastCost,
  resetPricingCache,
} from "./backends/classify.ts";

// ── backends/extraction ───────────────────────────────────────────────────────
export {
  ExtractionBackend,
  MockExtractionBackend,
  splitPdftext,
  PopplerBackend,
  CachingBackend,
  VisionBackend,
} from "./backends/extraction.ts";

// ── storage ───────────────────────────────────────────────────────────────────
export { initializeStore, insertChunks, sourceStats, similaritySearch, bm25Search } from "./storage.ts";

// ── ingest ────────────────────────────────────────────────────────────────────
export type { IngestOptions } from "./ingest.ts";
export { ingest, injectPageMarker, assignPages } from "./ingest.ts";
