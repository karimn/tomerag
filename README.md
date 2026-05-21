# TomeRAG

Bun/TypeScript RAG ingestion library for tabletop RPG content. Chunks markdown
or PDF rulebooks, classifies chunks by content type, embeds them with Ollama
or Nomic Atlas, and stores the result in a DuckDB file with HNSW vector search
and BM25 full-text search indices.

## Requirements

- [Bun](https://bun.sh) 1.3+
- For embeddings, **one of**:
  - [Ollama](https://ollama.com) running locally
  - A [Nomic Atlas](https://atlas.nomic.ai) API key (hosted, no local model)
  - `mock` mode (tests; no model needed)
- DuckDB extensions `vss` and `fts` are installed automatically on first run

## Installation

**From another repo (recommended):**

```bash
bun add github:karimn/tomerag
```

Then import from the package entry point:

```ts
import { ingest, initializeStore, OllamaBackend, HeuristicBackend } from "tomerag";
```

**As a git submodule:**

```bash
git submodule add https://github.com/karimn/tomerag tomerag
cd tomerag && bun install
```

```ts
import { ingest, initializeStore, OllamaBackend, HeuristicBackend } from "./tomerag/src/index.ts";
```

## Quick start

```ts
import { initializeStore, ingest, OllamaBackend, HeuristicBackend, PBTA_CONTENT_TYPES } from "tomerag";
import { defaultChunkingConfig } from "tomerag";
import type { Source } from "tomerag";

const source: Source = {
  id: "ironsworn",
  name: "Ironsworn",
  system: "PbtA",
  dbPath: "./ironsworn.duckdb",
  embeddingModel: "nomic-embed-text",
  embeddingDim: 768,
  license: "cc_by",
  chunking: defaultChunkingConfig(),        // minTokens:100, maxTokens:800, overlap:50
  contentTypes: new Set(PBTA_CONTENT_TYPES),
};

await initializeStore(source);

const inserted = await ingest({
  source,
  path: "./ironsworn-core.md",
  docId: "ironsworn-core",
  documentType: "core_rules",
  embedBackend: new OllamaBackend({ model: source.embeddingModel, dim: source.embeddingDim }),
  classifyBackend: new HeuristicBackend(),
});

console.log(`Inserted ${inserted} chunks into ${source.dbPath}`);
```

## Source config reference

A `Source` describes a single DuckDB database (one game system, one rulebook collection).

| Field | Type | Description |
|---|---|---|
| `id` | `string` | Unique identifier for this source |
| `name` | `string` | Human-readable name |
| `system` | `string` | Game system name (e.g. `"PbtA"`, `"Ironsworn"`) |
| `dbPath` | `string` | Absolute or relative path to the `.duckdb` file |
| `embeddingModel` | `string` | Ollama model name (e.g. `"nomic-embed-text"`) |
| `embeddingDim` | `number` | Embedding dimension (must match the model) |
| `license` | `License` | See license values below |
| `chunking` | `ChunkingConfig` | Optional — see chunking config below |
| `contentTypes` | `Set<ContentType>` | Which content types to classify into |

### License values

`cc_by` · `cc_by_sa` · `ogl` · `orc` · `homebrew` · `proprietary`

### Document type values

`core_rules` · `adventure` · `supplement` · `campaign`

### Chunking config

```ts
defaultChunkingConfig({
  minTokens: 100,       // minimum chunk size (whitespace tokens)
  maxTokens: 800,       // maximum chunk size
  overflow: "paragraph", // split strategy when over budget: "paragraph" | "sentence" | "token"
  overlapTokens: 50,    // tokens of overlap between consecutive chunks
})
```

All fields are optional — `defaultChunkingConfig()` with no arguments produces the defaults above.

## Content types

Use one of the preset sets or build your own `Set<ContentType>`:

| Export | Includes |
|---|---|
| `DEFAULT_CONTENT_TYPES` | `mechanic`, `lore`, `adventure_scene`, `table`, `stat_block`, `example`, `gm_guidance`, `flavor`, `procedure`, `boxed_text` |
| `PBTA_CONTENT_TYPES` | All of the above + `move`, `gm_move`, `playbook`, `oracle`, `front` |
| `YZE_CONTENT_TYPES` | `DEFAULT_CONTENT_TYPES` + `oracle`, `faction`, `gear` |

Full list of valid content type strings: `mechanic` · `lore` · `adventure_scene` · `table` · `stat_block` · `example` · `gm_guidance` · `flavor` · `procedure` · `boxed_text` · `move` · `gm_move` · `playbook` · `oracle` · `front` · `faction` · `gear`

## Embedding backends

### Ollama (production)

```ts
new OllamaBackend({
  model: "nomic-embed-text",  // any Ollama embedding model
  dim: 768,                    // must match model output dimension
  baseUrl: "http://localhost:11434",  // default
  batchSize: 32,               // texts per API call, default 32
})
```

Pull the model first: `ollama pull nomic-embed-text`

Common models and dimensions:

| Model | Dim |
|---|---|
| `nomic-embed-text` | 768 |
| `mxbai-embed-large` | 1024 |
| `all-minilm` | 384 |

### Nomic Atlas (hosted, no local model)

```ts
new NomicHostedBackend({
  apiKey: "nk-...",                    // optional; falls back to NOMIC_API_KEY / config
  model: "nomic-embed-text-v1.5",      // default; also the static DEFAULT_MODEL
  dim: 768,                            // default; also DEFAULT_DIM
  batchSize: 32,                       // default
})
```

The key is resolved in this order: explicit `apiKey` → `NOMIC_API_KEY` env var →
`nomicApiKey` in `tomerag.config.json`. Get a key from
<https://atlas.nomic.ai/account>.

```json
{ "nomicApiKey": "nk-..." }
```

Endpoint: `https://api-atlas.nomic.ai/v1/embedding/text`. The backend posts
`{ model, texts }` with `Authorization: Bearer <key>`.

### Mock (tests / no Ollama)

```ts
new MockEmbeddingBackend({ dim: 8 })  // hash-based, deterministic, not semantic
```

## Classification backends

### Heuristic (fast, no API)

Pattern-matches on headings and text to assign `contentType`. Good enough for
structured rulebooks.

```ts
new HeuristicBackend()
```

### Claude (accurate, requires API key)

Uses `claude-haiku` to classify batches of chunks. Costs money.

```ts
new ClaudeBackend({
  contentTypes: source.contentTypes,
  systemHint: source.system,   // e.g. "PbtA" — added to the system prompt
  model: "claude-haiku-4-5-20251001",  // default
  batchSize: 20,               // chunks per API call, default 20
  apiKey: "sk-ant-...",        // or set ANTHROPIC_API_KEY env var
})
```

Set `ANTHROPIC_API_KEY` in the environment or put it in `tomerag.config.json`:

```json
{ "anthropicApiKey": "sk-ant-..." }
```

### Cost estimation before ingesting

```ts
import { forecastCost, ClaudeBackend } from "tomerag";

const cost = await forecastCost(backend, rawChunks);
console.log(`${cost.nChunks} chunks, ${cost.nBatches} batches`);
console.log(`~$${cost.totalCostUsd.toFixed(4)} USD`);
```

## PDF ingestion

Requires `poppler-utils` installed (`apt install poppler-utils` / `brew install poppler`):

```ts
import { PopplerBackend, CachingBackend } from "tomerag";

const extractor = new CachingBackend(
  new PopplerBackend(),
  "./.cache/pdf",   // caches extracted text per file hash
);

await ingest({
  source, path: "./rulebook.pdf", docId: "rulebook",
  documentType: "core_rules",
  format: "pdf",
  embedBackend, classifyBackend,
  extractionBackend: extractor,
});
```

## Querying the database

The produced `.duckdb` file can be queried directly:

```sql
-- Vector similarity (top-k)
LOAD vss;
SELECT id, text, heading_path, content_type, move_trigger, page,
       array_cosine_similarity(embedding, [...]::FLOAT[768]) AS score
FROM chunks
ORDER BY score DESC LIMIT 5;

-- Full-text search
LOAD fts;
SELECT id, text, content_type,
       fts_main_chunks.match_bm25(id, 'your query') AS score
FROM chunks
WHERE fts_main_chunks.match_bm25(id, 'your query') IS NOT NULL
ORDER BY score DESC LIMIT 5;
```

`heading_path` is stored as a JSON-encoded string (`JSON.parse(row.heading_path)` → `string[]`).

## CLI

Create a source config JSON file:

```json
{
  "id": "ironsworn",
  "name": "Ironsworn",
  "system": "PbtA",
  "dbPath": "/absolute/path/to/ironsworn.duckdb",
  "embeddingModel": "nomic-embed-text",
  "embeddingDim": 768,
  "license": "cc_by",
  "chunking": { "minTokens": 50, "maxTokens": 600, "overflow": "paragraph", "overlapTokens": 50 },
  "contentTypes": ["mechanic", "move", "gm_move", "playbook", "lore", "table", "oracle"]
}
```

Then ingest:

```bash
bun run src/cli/ingest.ts \
  --source-config sources/ironsworn.json \
  --doc-id ironsworn-core \
  --document-type core_rules \
  --path rulebooks/ironsworn.md \
  --classify heuristic \
  --embed ollama
```

`--classify` defaults to `heuristic`; `--embed` defaults to `ollama` and also
accepts `nomic` or `mock`. For `nomic`, the key comes from `--nomic-api-key`,
`NOMIC_API_KEY`, or `tomerag.config.json`:

```bash
bun run src/cli/ingest.ts \
  --source-config sources/ironsworn.json \
  --doc-id ironsworn-core \
  --document-type core_rules \
  --path rulebooks/ironsworn.md \
  --embed nomic
```

The CLI passes `embeddingModel` / `embeddingDim` from the source config straight
through, so set those to `"nomic-embed-text-v1.5"` / `768` in your source JSON
when using Nomic.

## Re-ingesting / deduplication

`insertChunks` deduplicates by `(doc_id, content_hash)`. Re-ingesting the same
file returns `0` inserted. To replace a document, delete its chunks first:

```ts
import { DuckDBInstance } from "@duckdb/node-api";
const inst = await DuckDBInstance.create(source.dbPath);
const conn = await inst.connect();
await conn.run("DELETE FROM chunks WHERE doc_id = ?", ["ironsworn-core"]);
conn.closeSync();
```

Then re-ingest normally.

### Migrating from Ollama `nomic-embed-text` → Nomic Atlas `nomic-embed-text-v1.5`

These are **different model versions**: cosine similarity between embeddings
from the two is unreliable, so a mixed-model database silently degrades search
quality. Detect mixed-model databases and re-ingest before querying.

Detect what's in an existing DB:

```sql
SELECT embedding_model, COUNT(*) FROM chunks GROUP BY embedding_model;
```

If you see `nomic-embed-text` (unversioned, from local Ollama) alongside or
instead of `nomic-embed-text-v1.5`, you need to re-ingest. The vector dimension
(`768`) is the same across versions, so the DuckDB schema doesn't change.

To migrate a single source:

1. Update the source config's `embeddingModel` to `"nomic-embed-text-v1.5"`.
2. Delete the existing rows (or the whole `.duckdb` file):

   ```ts
   await conn.run("DELETE FROM chunks WHERE source_id = ?", [source.id]);
   await conn.run("DELETE FROM source_meta WHERE id = ?", [source.id]);
   ```

3. Re-run ingestion with `--embed nomic`. The new `source_meta.embedding_model`
   will read `nomic-embed-text-v1.5` and all chunks will be consistent.

## Development

```bash
bun install
bun test                          # unit + schema-contract tests (no network required)
TOMERAG_LIVE_TESTS=1 bun test     # also runs Ollama/Claude/Poppler live tests
bun run typecheck
```

See `docs/superpowers/specs/2026-04-30-bun-ts-port-design.md` for the full design.
