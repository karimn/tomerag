# TomeRAG Bun/TS Port — Design Spec

**Status:** Draft
**Date:** 2026-04-30
**Goal:** Port the Julia TomeRAG ingestion library to Bun/TypeScript while preserving the DuckDB schema consumed by the existing `scribe/` plugin in `agentic-ironsworn`.

## Context

`TomeRAG.jl` is a ~3.4k-LOC Julia RAG library for tabletop RPG content. It does two things today: (1) ingest markdown/PDF sources into per-system DuckDB files, and (2) expose an Oxygen-based REST API over that data.

The query/server surface has already been migrated to TypeScript inside `agentic-ironsworn/plugins/ironsworn/scribe/src/rag/`. What remains in Julia is the **ingestion pipeline** that produces `ironsworn.duckdb`. This spec covers porting the ingestion side only; the scribe read-path is unchanged.

## Motivation

- **Deployment / ecosystem.** Bun/TS ships easily, integrates cleanly with `scribe/`, avoids a cross-runtime boundary for anyone touching both sides.
- **Startup time.** Julia's JIT warmup is painful for the short-lived, script-style workflow ingestion is.

## Non-goals

- Porting the REST server (`src/server.jl`) — already replaced by `scribe/`.
- Porting the `scripts/extract_structured.jl` batch script — separate data-extraction tool, lives outside the library.
- Schema redesign — existing DuckDB layout is a frozen contract with the scribe reader.

## Scope

Full parity with the current Julia library surface, minus the REST server:

- Types: `Chunk`, `Source`, `ChunkingConfig`, `QueryResult`, `RawChunk`, `PageText`, content-type sets (default, PbtA, YZE).
- Embedding backends: `MockEmbeddingBackend`, `OllamaBackend`.
- Classify backends: `MockClassifyBackend`, `HeuristicBackend`, `ClaudeBackend` (with batch API, fallback to heuristic on failure), `forecastCost()` + `CostEstimate`.
- Extraction backends: `MockExtractionBackend`, `PopplerBackend` (shelling to `pdftotext`), `CachingBackend`, `VisionBackend` (Claude Haiku + `pdftoppm`).
- Chunker: heading-aware markdown parser, token-budget splitter with paragraph → sentence → token overflow ladder, atomic markdown-table handling, `heading_path` prepended for search.
- Storage: `initializeStore`, `insertChunks`, `sourceStats`, plus `similaritySearch` and `bm25Search` for round-trip testing.
- Ingest orchestration: `ingest()` covers markdown + PDF, including page marker injection and `_assignPages`.
- CLI: `bun run src/cli/ingest.ts` wrapper for one-shot ingestion runs.

## Decisions

| Topic | Decision |
|---|---|
| Runtime | Bun-first, Node-unchecked. Use Bun APIs freely (Bun.file, Bun test). |
| Repo | Rename the current `tomerag-jl/` directory to `tomerag/` in place; remove Julia files; same `.git`. |
| Schema | Byte-compatible with existing `chunks` / `source_meta` tables, HNSW + FTS indices, JSON-encoded `heading_path` / `tags`. No migration required for the scribe reader. |
| Config | `./tomerag.config.json` in cwd, validated with zod. No env-var fallback. Gitignored. |
| Dispatch | Abstract classes with subclasses (not discriminated unions). One file per backend domain. |
| Symbols | Julia `Symbol`s (`:move`, `:pdf`) → TypeScript string-literal unions. |
| Async | All backend operations are `Promise`-returning, including mock backends (interface uniformity). |
| Registry | Drop `SourceRegistry`. Ingestion operates on a single `Source` per call. |
| Anthropic SDK | `@anthropic-ai/sdk` for classify + vision. |
| DuckDB | `@duckdb/node-api` (same as `scribe/`, confirms Bun compatibility). |

## Module layout

```
tomerag/
  package.json                  # type: module, bun-first
  tsconfig.json
  bun.lock
  tomerag.config.example.json   # template (gitignored real file)
  README.md
  src/
    index.ts                    # public re-exports
    types.ts                    # Chunk, QueryResult, Source, ChunkingConfig, RawChunk, PageText, enums
    content-types.ts            # DEFAULT_CONTENT_TYPES, PBTA_CONTENT_TYPES, YZE_CONTENT_TYPES
    tokenize.ts                 # normalizeText, tokenCount, contentHash
    chunker.ts                  # parseMarkdownSections, splitToTokenBudget, chunkDocument
    storage.ts                  # initializeStore, insertChunks, sourceStats, similaritySearch, bm25Search
    ingest.ts                   # ingest() orchestration + assignPages + injectPageMarker
    config.ts                   # loadConfig() -> { anthropicApiKey }
    backends/
      embedding.ts              # EmbeddingBackend (abstract) + Mock + Ollama
      classify.ts               # ClassifyBackend (abstract) + Mock + Heuristic + Claude + forecastCost + CostEstimate
      extraction.ts             # ExtractionBackend (abstract) + Mock + Poppler + Caching + Vision
    cli/
      ingest.ts                 # argv parser + ingestion runner
  test/
    chunker.test.ts
    tokenize.test.ts
    types.test.ts
    content-types.test.ts
    embedding.test.ts
    classify.test.ts
    heuristic.test.ts
    extraction.test.ts
    storage.test.ts
    ingest.test.ts
    integration.test.ts         # byte-compat: TS ingest → scribe reader
    fixtures/
      ironsworn_sample.md
```

## Data model

### Schema contract (frozen)

The `chunks` table matches the Julia version column-for-column:

```sql
CREATE TABLE chunks (
  id               TEXT PRIMARY KEY,
  source_id        TEXT,
  doc_id           TEXT,
  doc_path         TEXT,
  text             TEXT,
  embedding        FLOAT[<embedding_dim>],
  embedding_model  TEXT,
  token_count      INTEGER,
  content_hash     TEXT,
  document_type    TEXT,
  system           TEXT,
  edition          TEXT,
  page             TEXT,
  heading_path     TEXT,       -- JSON-encoded string[]
  chunk_order      INTEGER,
  parent_id        TEXT,
  content_type     TEXT,
  tags             TEXT,       -- JSON-encoded string[]
  move_trigger     TEXT,
  scene_type       TEXT,
  encounter_key    TEXT,
  npc_name         TEXT,
  license          TEXT
);

CREATE UNIQUE INDEX chunks_dedup_idx ON chunks(doc_id, content_hash);
CREATE INDEX chunks_hnsw_idx ON chunks USING HNSW (embedding) WITH (metric='cosine');
PRAGMA create_fts_index('chunks', 'id', 'text', stemmer='porter', overwrite=1);
```

`source_meta` keys: `source_id`, `embedding_model`, `embedding_dim`, `system`.

`scribe/src/rag/query.ts` reads `id, text, heading_path, content_type, move_trigger, page, embedding` and calls `array_cosine_similarity` + `fts_main_chunks.match_bm25`. All of these stay identical.

### Core types

```typescript
export type License = 'cc_by' | 'cc_by_sa' | 'ogl' | 'orc' | 'homebrew' | 'proprietary';
export type DocumentType = 'core_rules' | 'adventure' | 'supplement' | 'campaign';
export type ContentType =
  | 'mechanic' | 'lore' | 'adventure_scene' | 'table' | 'stat_block'
  | 'example' | 'gm_guidance' | 'flavor' | 'procedure' | 'boxed_text'
  | 'move' | 'gm_move' | 'playbook' | 'oracle' | 'front'
  | 'faction' | 'gear';
export type Overflow = 'paragraph' | 'sentence' | 'token';

export interface ChunkingConfig {
  minTokens: number;       // default 100
  maxTokens: number;       // default 800
  overflow: Overflow;      // default 'paragraph'
  overlapTokens: number;   // default 50
  atomicPatterns: RegExp[];
}

export interface Chunk {
  id: string;
  sourceId: string;
  docId: string;
  docPath: string;
  text: string;
  embedding: Float32Array;
  embeddingModel: string;
  tokenCount: number;
  contentHash: string;
  documentType: DocumentType;
  system: string;
  edition: string;
  page: string;
  headingPath: string[];
  chunkOrder: number;
  parentId: string | null;
  contentType: ContentType;
  tags: string[];
  moveTrigger: string | null;
  sceneType: string | null;
  encounterKey: string | null;
  npcName: string | null;
  license: License;
}

export interface QueryResult { chunk: Chunk; score: number; rank: number; }

export interface Source {
  id: string;
  name: string;
  system: string;
  dbPath: string;
  embeddingModel: string;
  embeddingDim: number;
  license: License;
  chunking: ChunkingConfig;
  contentTypes: Set<ContentType>;
}

export interface RawChunk {
  headingPath: string[];
  text: string;
  chunkOrder: number;
  page: string;          // '' for markdown; populated from page markers for PDF
}

export interface PageText { pageNum: number; text: string; }
```

TypeScript uses `camelCase` at the type level. SQL column names stay `snake_case` on the wire. Conversion happens in a `rowToChunk(row)` helper.

## Backend interfaces

Dispatch is done via abstract classes with concrete subclasses — mirrors the object-oriented style used in `scribe/` and gives clean constructor-injected config.

### Embedding

```typescript
export abstract class EmbeddingBackend {
  abstract embed(text: string): Promise<Float32Array>;
  abstract embed(texts: string[]): Promise<Float32Array[]>;
}

export class MockEmbeddingBackend extends EmbeddingBackend {
  constructor(opts?: { dim?: number });   // default dim=8, deterministic SHA-256 derived
}

export class OllamaBackend extends EmbeddingBackend {
  constructor(opts: {
    model: string;
    dim: number;
    baseUrl?: string;         // default 'http://localhost:11434'
    batchSize?: number;       // default 32
  });
}
```

Ollama uses `fetch` against `/api/embed`; response validated with zod (`{ embeddings: number[][] }`). Batches iterate in `batchSize` chunks.

### Classify

```typescript
export interface Classification {
  contentType: ContentType;
  tags: string[];
  moveTrigger: string | null;
  sceneType: string | null;
  encounterKey: string | null;
  npcName: string | null;
}

export abstract class ClassifyBackend {
  abstract classify(input: { text: string; headingPath: string[] }): Promise<Classification>;
  async classifyBatch(raws: RawChunk[]): Promise<Classification[]> {
    // Default: map sequentially. Claude overrides for batched API.
  }
}

export class MockClassifyBackend extends ClassifyBackend {
  constructor(opts?: { contentType?: ContentType; tags?: string[] });
}

export class HeuristicBackend extends ClassifyBackend {
  // Regex patterns for heading-based classification. Port the Julia logic verbatim:
  // 1. **When ...** trigger → :move
  // 2. multiple pipe-rows or oracle heading → :table
  // 3. bestiary heading + stat-line → :stat_block
  // 4. gm guidance heading → :gm_guidance
  // 5. lore heading → :lore
  // 6. fallback → :mechanic
}

export class ClaudeBackend extends ClassifyBackend {
  constructor(opts: {
    contentTypes: Set<ContentType>;       // required
    apiKey?: string;                       // defaults to loadConfig().anthropicApiKey
    model?: string;                        // default 'claude-haiku-4-5-20251001'
    batchSize?: number;                    // default 20
    systemHint?: string;                   // e.g. 'PbtA'
  });
  // Overrides classifyBatch: builds prompt with N chunks, extracts JSON array
  // (bracket-counting, handles code fences + commentary), falls back to
  // HeuristicBackend per-chunk on any batch failure or item-count mismatch.
}

export interface CostEstimate {
  model: string;
  nChunks: number;
  nBatches: number;
  inputTokens: number;
  outputTokens: number;
  totalCostUsd: number;
}

export function forecastCost(
  backend: ClaudeBackend,
  raws: RawChunk[],
): Promise<CostEstimate>;
```

`forecastCost` calls Anthropic's `count_tokens` endpoint per batch, estimates output at 50 tokens/chunk, and fetches per-token pricing from the LiteLLM GitHub JSON (cached per process; falls back to Sonnet rates on fetch failure).

### Extraction

```typescript
export abstract class ExtractionBackend {
  abstract extractPages(pdfPath: string): Promise<PageText[]>;
  async extractPage(pdfPath: string, pageNum: number): Promise<PageText> {
    // Default: extractPages then find. Poppler/Vision override.
  }
}

export class MockExtractionBackend extends ExtractionBackend {
  constructor(pages: PageText[]);
}

export class PopplerBackend extends ExtractionBackend {
  // Shells to `pdftotext -layout`. Splits on '\f'. extractPage overrides with -f/-l.
}

export class CachingBackend extends ExtractionBackend {
  constructor(opts: { inner: ExtractionBackend; cacheDir: string });
  // Hash PDF bytes with SHA-256, cache as <cacheDir>/<hash>/page_NNN.txt.
}

export class VisionBackend extends ExtractionBackend {
  constructor(opts?: {
    apiKey?: string;                       // defaults to loadConfig().anthropicApiKey
    model?: string;                        // default 'claude-haiku-4-5-20251001'
    concurrency?: number;                  // default 5
    dpi?: number;                          // default 150
  });
  // Renders page with pdftoppm, base64-encodes, POSTs to /v1/messages.
}
```

Concurrency for `VisionBackend.extractPages`: inline 10-line `pLimit` helper (no external dep).

## Ingestion pipeline

```typescript
export interface IngestOptions {
  source: Source;
  path: string;
  docId: string;
  documentType: DocumentType;
  format?: 'auto' | 'markdown' | 'pdf';     // default 'auto' (by extension)
  embedBackend: EmbeddingBackend;
  classifyBackend: ClassifyBackend;
  extractionBackend?: ExtractionBackend;    // required when format resolves to 'pdf'
}

export async function ingest(opts: IngestOptions): Promise<number>;
```

Flow (1:1 with Julia `ingest!`):

1. Resolve `format` from extension (`.pdf` → `'pdf'`, else `'markdown'`) if `'auto'`.
2. Error if `format === 'pdf'` and `extractionBackend` is missing.
3. Load text:
   - markdown: `Bun.file(path).text()`.
   - pdf: `extractionBackend.extractPages()`, inject `<!-- page N -->` markers. Marker goes on line 2 if page starts with a heading (so it lands in section body not discarded preamble), else prepended.
4. `chunkDocument(text, source.chunking)` → `RawChunk[]`.
5. For pdf only: `assignPages(raws)` — scan markers, strip them, set each chunk's `.page` from the last marker it contained.
6. `classified = await classifyBackend.classifyBatch(raws)`.
7. Build search texts: prepend `headingPath.join(' > ') + '\n\n'` if heading path non-empty. Same text goes into embedding and stored `chunk.text` (so BM25 and dense both find by heading name).
8. `embeddings = await embedBackend.embed(searchTexts)`.
9. Assemble `Chunk[]` with `crypto.randomUUID()` ids.
10. `return insertChunks(source, chunks)`.

Preserved invariants:

- `contentHash = sha256(normalizeText(rawText))`; `normalizeText` = lowercase + collapse whitespace.
- `tokenCount = text.split(/\s+/).length` — not a real tokenizer; matches Julia.
- Atomic markdown tables in the token-cap splitter (tables never split even if they exceed `maxTokens` alone).
- Dedup scoped to `(doc_id, content_hash)` via unique index.

## Storage

```typescript
export async function initializeStore(source: Source): Promise<void>;
export async function insertChunks(source: Source, chunks: Chunk[]): Promise<number>;
export async function sourceStats(source: Source): Promise<{
  chunkCount: number;
  embeddingModel: string;
  embeddingDim: number;
}>;
export async function similaritySearch(
  source: Source,
  queryEmbedding: Float32Array,
  opts?: { topK?: number; filters?: Partial<Record<'content_type' | 'document_type' | 'system' | 'doc_id', string>> },
): Promise<QueryResult[]>;
export async function bm25Search(
  source: Source,
  queryText: string,
  opts?: { topK?: number; filters?: Partial<Record<'content_type' | 'document_type' | 'system' | 'doc_id', string>> },
): Promise<Array<{ chunk: Chunk; score: number }>>;
```

`insertChunks` is transactional, does per-chunk SELECT-then-INSERT dedup on `(doc_id, content_hash)`, returns count actually inserted. On any inserts: drop and recreate the FTS index (HNSW auto-updates with `hnsw_enable_experimental_persistence = true`).

**DuckDB binding details:**
- `conn.run(sql, params)` for writes, `conn.runAndReadAll(sql, params)` for reads.
- Embedding values embedded as SQL literal (`[${arr.join(',')}]::FLOAT[${dim}]`) not a bind param — matches what `scribe/` does, same reason (driver param-binding issues with FLOAT[] arrays).
- `SET hnsw_enable_experimental_persistence = true;` on every connection.

Search helpers live in `storage.ts` so ingestion tests can round-trip ingest → search → assert. They're exported but the primary consumer is the scribe reader (which doesn't use them — it queries DuckDB directly).

## CLI

```bash
bun run src/cli/ingest.ts \
  --source-config sources/ironsworn.json \
  --doc-id ironsworn-core \
  --document-type core_rules \
  --path rulebooks/ironsworn.md \
  [--classify heuristic|claude] \
  [--embed ollama|mock]
```

- `--source-config`: JSON file matching the `Source` shape (`id`, `name`, `system`, `dbPath`, `embeddingModel`, `embeddingDim`, `license`, `chunking`, `contentTypes`).
- Defaults: `--embed ollama`, `--classify heuristic`.
- Uses `parseArgs` from `node:util` — no CLI dep.
- Prints chunked/classified/embedded/inserted counts.

## Config

`./tomerag.config.json` schema:

```typescript
const ConfigSchema = z.object({
  anthropicApiKey: z.string().min(1).optional(),
});
```

Loaded by `loadConfig()`, cached module-local. Missing file or missing key → error message pointing the user at the expected path and field. No env-var fallback.

Gitignored by default; repo ships `tomerag.config.example.json` as a template.

## Testing

- Bun built-in test runner (`bun test`).
- One `*.test.ts` per Julia `test_*.jl` file.
- All unit tests use mock backends (no network, no PDF tools, no API key).
- Live tests guarded by the `TOMERAG_LIVE_TESTS=1` environment variable (test-only flag — not a library config mechanism): Ollama embeddings, Claude classify, Poppler on a real PDF, Vision on a real PDF. Skipped otherwise.
- Byte-compat golden test (`integration.test.ts`): ingest `fixtures/ironsworn_sample.md`, then dynamically import `scribe/src/rag/query.ts` against the resulting DuckDB — asserts chunks come back with expected shape.

## Migration

1. In-place rename: `mv tomerag-jl tomerag`.
2. Remove Julia files: `src/*.jl`, `test/*.jl`, `Project.toml`, `Manifest.toml`. Keep the `test/fixtures/` directory. Replace `.gitignore` with a TS-appropriate one (ignore `node_modules/`, `tomerag.config.json`, `dist/`, `*.duckdb`, `.DS_Store`).
3. Initialize Bun project: `bun init`; add `@duckdb/node-api`, `@anthropic-ai/sdk`, `zod` as deps, `typescript`, `@types/bun` as devDeps. UUIDs come from built-in `crypto.randomUUID()` — no dep needed.
4. Port in order (each a plan step): types → tokenize → chunker → backends/embedding → backends/classify → backends/extraction → storage → ingest → CLI → integration test.
5. After each port step, port the matching Julia test file and get it passing.
6. Final check: run `integration.test.ts` that round-trips through the real `scribe/src/rag/query.ts`.

## Open questions

None blocking. Two things to verify during implementation (not design decisions):

- Exact form of `@duckdb/node-api`'s Float32Array → `FLOAT[N]` binding. Fallback is SQL-literal embedding (what `scribe/` does).
- Anthropic SDK v3+ batch API shape for `messages.countTokens` — method name may be `count_tokens` or `countTokens` depending on version.
