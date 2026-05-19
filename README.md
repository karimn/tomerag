# TomeRAG

Bun/TypeScript RAG ingestion library for tabletop RPG content. Ports the
`TomeRAG.jl` ingestion pipeline; produces a DuckDB schema consumed unchanged by
the `scribe/` reader.

## Setup

```bash
bun install
cp tomerag.config.example.json tomerag.config.json   # add your Anthropic key (optional)
```

## Test

```bash
bun test                  # unit + schema-contract tests (no network/keys)
TOMERAG_LIVE_TESTS=1 bun test   # also runs Ollama/Claude/Poppler/Vision live tests
bun run typecheck
```

## Ingest (CLI)

```bash
bun run src/cli/ingest.ts \
  --source-config sources/ironsworn.json \
  --doc-id ironsworn-core \
  --document-type core_rules \
  --path rulebooks/ironsworn.md \
  [--classify heuristic|claude] \
  [--embed ollama|mock]
```

## Library

```ts
import { ingest, initializeStore, MockEmbeddingBackend, HeuristicBackend } from "./src/index.ts";
```

See `docs/superpowers/specs/2026-04-30-bun-ts-port-design.md` for the design.
