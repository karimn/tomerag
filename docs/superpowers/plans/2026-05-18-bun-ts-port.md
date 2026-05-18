# TomeRAG Bun/TS Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the Julia `TomeRAG.jl` ingestion library to Bun/TypeScript, producing a byte-compatible `chunks`/`source_meta` DuckDB schema consumed unchanged by the `scribe/` reader.

**Architecture:** One-file-per-domain modules under `src/`. Backends are abstract classes with concrete subclasses, constructor-injected config. All backend ops are `Promise`-returning. DuckDB via `@duckdb/node-api` (embeddings written as SQL literals, matching `scribe/`). Anthropic via `@anthropic-ai/sdk` v0.39. TDD: one `*.test.ts` per Julia `test_*.jl`, mock backends only in unit tests, live tests gated by `TOMERAG_LIVE_TESTS=1`.

**Tech Stack:** Bun 1.3+, TypeScript 5.8, `@duckdb/node-api` 1.5.2-r.1, `@anthropic-ai/sdk` 0.39.0, `zod` 4. Built-in `bun:test`, `node:crypto`, `node:util`, `node:path`, `crypto.randomUUID()`.

**Reference (read-only, do not modify):** Julia sources in `src/*.jl` and tests in `test/*.jl` are the behavioral spec. The frozen schema consumer is `/media/karim/Code-Drive/karimn-code/rpg-rules/plugins/ironsworn/scribe/src/rag/query.ts` — it reads `id, text, heading_path, content_type, move_trigger, page, embedding` and calls `array_cosine_similarity` + `fts_main_chunks.match_bm25`.

**Key porting invariants (apply to every task):**
- `wsTokens(s)` = whitespace token split that returns `[]` for empty/blank input (Julia `split("")` → `[]`, but JS `"".split(/\s+/)` → `[""]`; the helper must special-case this). All token counting and budgeting goes through it.
- TypeScript types are `camelCase`; SQL columns stay `snake_case`; conversion only in `rowToChunk`.
- `contentHash`/`tokenCount` are computed from the raw section body (`RawChunk.text`), **not** the heading-prefixed search text that gets stored in `chunk.text`.
- Julia `Symbol`s → string-literal unions. `nothing` → `null`. `Set{Symbol}` → `Set<ContentType>`.
- Embeddings are written to DuckDB as the SQL literal `[${Array.from(emb).join(",")}]::FLOAT[${dim}]`, never as a bind parameter (matches `scribe/`, avoids `FLOAT[]` driver binding issues). All other values use `?` positional bind params.

---

## File Structure

| File | Responsibility |
|---|---|
| `package.json` | `type: module`, scripts, deps |
| `tsconfig.json` | Bun-targeted strict TS |
| `.gitignore` | node_modules, config, dist, *.duckdb |
| `tomerag.config.example.json` | config template |
| `src/types.ts` | `License`, `DocumentType`, `ContentType`, `Overflow`, `ChunkingConfig`, `Chunk`, `QueryResult`, `Source`, `RawChunk`, `PageText`, `Section`, `Classification` |
| `src/content-types.ts` | `DEFAULT_CONTENT_TYPES`, `PBTA_CONTENT_TYPES`, `YZE_CONTENT_TYPES` |
| `src/tokenize.ts` | `normalizeText`, `tokenCount`, `contentHash`, `wsTokens`, `wsCount` |
| `src/chunker.ts` | `parseMarkdownSections`, `splitToTokenBudget`, `chunkDocument` |
| `src/config.ts` | `loadConfig`, `resetConfigCache` (test helper) |
| `src/backends/embedding.ts` | `EmbeddingBackend`, `MockEmbeddingBackend`, `OllamaBackend` |
| `src/backends/classify.ts` | `ClassifyBackend`, `Mock`/`Heuristic`/`ClaudeBackend`, `CostEstimate`, `forecastCost` |
| `src/backends/extraction.ts` | `ExtractionBackend`, `Mock`/`Poppler`/`Caching`/`VisionBackend` |
| `src/storage.ts` | `initializeStore`, `insertChunks`, `sourceStats`, `similaritySearch`, `bm25Search`, `rowToChunk` |
| `src/ingest.ts` | `ingest`, `injectPageMarker`, `assignPages` |
| `src/cli/ingest.ts` | argv parser + runner |
| `src/index.ts` | public re-exports |
| `test/*.test.ts` | one per Julia `test_*.jl` |
| `test/fixtures/ironsworn_sample.md` | kept from Julia tree |

---

## Task 0: Migration & project scaffold

**Files:**
- Delete: `src/*.jl`, `test/*.jl`, `Project.toml`, `Manifest.toml`
- Create: `package.json`, `tsconfig.json`, `.gitignore`, `tomerag.config.example.json`
- Keep: `test/fixtures/ironsworn_sample.md`, `docs/`, `.git/`, `.remember/`, `README.md` (rewritten in Task 14)

> **DESTRUCTIVE — confirm with the user before running this task.** It removes all Julia source/tests and (optionally) renames the working directory. The spec's decision is an in-place rename `tomerag-jl/` → `tomerag/`. Renaming the cwd mid-session breaks relative paths and the second additional working directory (`/home/karim/Code/tomerag/tomerag-jl`). **Recommendation:** do the rename as the very last manual step (or skip it — the directory name is cosmetic and the git repo is unaffected). All tasks below assume the repo root is the current working directory regardless of its name.

- [ ] **Step 1: Remove Julia files (keep fixtures)**

Run:
```bash
git rm -q src/TomeRAG.jl src/backends.jl src/chunker.jl src/content_types.jl src/extraction.jl src/ingest.jl src/query_engine.jl src/server.jl src/storage.jl src/tokenize.jl src/types.jl
git rm -q test/runtests.jl test/test_backends.jl test/test_chunker.jl test/test_content_types.jl test/test_extraction.jl test/test_heuristic.jl test/test_ingest.jl test/test_integration.jl test/test_ollama.jl test/test_query.jl test/test_server.jl test/test_splitter.jl test/test_storage.jl test/test_tokenize.jl test/test_types.jl
git rm -q Project.toml Manifest.toml
```
Expected: files staged for deletion; `test/fixtures/ironsworn_sample.md` untouched.

- [ ] **Step 2: Write `package.json`**

```json
{
  "name": "tomerag",
  "version": "0.1.0",
  "type": "module",
  "private": true,
  "module": "src/index.ts",
  "scripts": {
    "test": "bun test",
    "typecheck": "tsc --noEmit",
    "ingest": "bun run src/cli/ingest.ts"
  },
  "dependencies": {
    "@anthropic-ai/sdk": "0.39.0",
    "@duckdb/node-api": "1.5.2-r.1",
    "zod": "^4.3.6"
  },
  "devDependencies": {
    "@types/bun": "^1.2.0",
    "typescript": "^5.8.0"
  }
}
```

- [ ] **Step 3: Write `tsconfig.json`**

```json
{
  "compilerOptions": {
    "lib": ["ESNext"],
    "module": "ESNext",
    "target": "ESNext",
    "moduleResolution": "bundler",
    "types": ["bun"],
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "noImplicitOverride": true,
    "skipLibCheck": true,
    "noEmit": true,
    "verbatimModuleSyntax": true
  },
  "include": ["src", "test"]
}
```

- [ ] **Step 4: Write `.gitignore`**

```gitignore
node_modules/
dist/
tomerag.config.json
*.duckdb
*.duckdb.wal
.DS_Store
```

- [ ] **Step 5: Write `tomerag.config.example.json`**

```json
{
  "anthropicApiKey": "sk-ant-..."
}
```

- [ ] **Step 6: Install dependencies**

Run: `bun install`
Expected: `bun.lock` created, `node_modules/` populated, no errors.

- [ ] **Step 7: Verify Bun test runner works**

Run: `bun test 2>&1 | tail -5`
Expected: `0 pass 0 fail` (no test files yet) — confirms the runner is wired.

- [ ] **Step 8: Commit**

```bash
git add package.json tsconfig.json .gitignore tomerag.config.example.json bun.lock
git commit -m "chore: scaffold Bun/TS project, remove Julia sources"
```

---

## Task 1: Core types + content types

**Files:**
- Create: `src/types.ts`, `src/content-types.ts`
- Test: `test/types.test.ts`, `test/content-types.test.ts`

- [ ] **Step 1: Write `test/types.test.ts`** (ports `test_types.jl` minus `SourceRegistry`, which the spec drops)

```typescript
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bun test test/types.test.ts`
Expected: FAIL — `Cannot find module "../src/types.ts"`.

- [ ] **Step 3: Write `src/types.ts`**

```typescript
export type License =
  | "cc_by" | "cc_by_sa" | "ogl" | "orc" | "homebrew" | "proprietary";

export type DocumentType =
  | "core_rules" | "adventure" | "supplement" | "campaign";

export type ContentType =
  | "mechanic" | "lore" | "adventure_scene" | "table" | "stat_block"
  | "example" | "gm_guidance" | "flavor" | "procedure" | "boxed_text"
  | "move" | "gm_move" | "playbook" | "oracle" | "front"
  | "faction" | "gear";

export type Overflow = "paragraph" | "sentence" | "token";

export interface ChunkingConfig {
  minTokens: number;
  maxTokens: number;
  overflow: Overflow;
  overlapTokens: number;
  atomicPatterns: RegExp[]; // present for parity with Julia; currently unused by the chunker (table-atomicity is hardcoded)
}

export function defaultChunkingConfig(
  overrides: Partial<ChunkingConfig> = {},
): ChunkingConfig {
  return {
    minTokens: 100,
    maxTokens: 800,
    overflow: "paragraph",
    overlapTokens: 50,
    atomicPatterns: [],
    ...overrides,
  };
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

export interface QueryResult {
  chunk: Chunk;
  score: number;
  rank: number;
}

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
  page: string; // "" for markdown; populated from page markers for PDF
}

export interface PageText {
  pageNum: number;
  text: string;
}

export interface Section {
  headingPath: string[];
  text: string;
}

export interface Classification {
  contentType: ContentType;
  tags: string[];
  moveTrigger: string | null;
  sceneType: string | null;
  encounterKey: string | null;
  npcName: string | null;
}
```

- [ ] **Step 4: Write `test/content-types.test.ts`** (ports `test_content_types.jl`)

```typescript
import { test, expect } from "bun:test";
import {
  DEFAULT_CONTENT_TYPES, PBTA_CONTENT_TYPES, YZE_CONTENT_TYPES,
} from "../src/content-types.ts";

test("content type sets", () => {
  for (const t of ["mechanic", "lore", "procedure", "boxed_text"] as const) {
    expect(DEFAULT_CONTENT_TYPES.has(t)).toBe(true);
  }
  expect(PBTA_CONTENT_TYPES.has("move")).toBe(true);
  expect(PBTA_CONTENT_TYPES.has("playbook")).toBe(true);
  for (const t of DEFAULT_CONTENT_TYPES) expect(PBTA_CONTENT_TYPES.has(t)).toBe(true);

  expect(YZE_CONTENT_TYPES.has("faction")).toBe(true);
  expect(YZE_CONTENT_TYPES.has("gear")).toBe(true);
  for (const t of DEFAULT_CONTENT_TYPES) expect(YZE_CONTENT_TYPES.has(t)).toBe(true);
});
```

- [ ] **Step 5: Write `src/content-types.ts`**

```typescript
import type { ContentType } from "./types.ts";

export const DEFAULT_CONTENT_TYPES: ReadonlySet<ContentType> = new Set<ContentType>([
  "mechanic", "lore", "adventure_scene", "table", "stat_block",
  "example", "gm_guidance", "flavor", "procedure", "boxed_text",
]);

export const PBTA_CONTENT_TYPES: ReadonlySet<ContentType> = new Set<ContentType>([
  ...DEFAULT_CONTENT_TYPES,
  "move", "gm_move", "playbook", "oracle", "front",
]);

export const YZE_CONTENT_TYPES: ReadonlySet<ContentType> = new Set<ContentType>([
  ...DEFAULT_CONTENT_TYPES,
  "oracle", "faction", "gear",
]);
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `bun test test/types.test.ts test/content-types.test.ts`
Expected: PASS (all assertions green).

- [ ] **Step 7: Commit**

```bash
git add src/types.ts src/content-types.ts test/types.test.ts test/content-types.test.ts
git commit -m "feat: port core types and content-type sets"
```

---

## Task 2: Tokenize

**Files:**
- Create: `src/tokenize.ts`
- Test: `test/tokenize.test.ts`

- [ ] **Step 1: Write `test/tokenize.test.ts`** (ports `test_tokenize.jl`)

```typescript
import { test, expect } from "bun:test";
import { tokenCount, normalizeText, contentHash } from "../src/tokenize.ts";

test("tokenize", () => {
  expect(tokenCount("hello world")).toBe(2);
  expect(tokenCount("  hello   world \n foo ")).toBe(3);
  expect(tokenCount("")).toBe(0);

  expect(normalizeText(" Hello\tWorld \n")).toBe("hello world");

  const h1 = contentHash("Hello World");
  const h2 = contentHash("hello world");
  const h3 = contentHash("hello  world");
  expect(h1).toBe(h2);
  expect(h2).toBe(h3);
  expect(h1.length).toBe(64);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bun test test/tokenize.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `src/tokenize.ts`**

```typescript
import { createHash } from "node:crypto";

/** Whitespace token split. Returns [] for empty/blank input (matches Julia `split("")`). */
export function wsTokens(s: string): string[] {
  const t = s.trim();
  return t === "" ? [] : t.split(/\s+/);
}

export function wsCount(s: string): number {
  return wsTokens(s).length;
}

/** Lowercase, trim, collapse internal whitespace. Used only for dedup hashing. */
export function normalizeText(s: string): string {
  return s.toLowerCase().trim().replace(/\s+/g, " ");
}

export function tokenCount(s: string): number {
  return wsCount(s);
}

/** SHA-256 of the normalized text, lowercase hex. */
export function contentHash(s: string): string {
  return createHash("sha256").update(normalizeText(s), "utf8").digest("hex");
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `bun test test/tokenize.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tokenize.ts test/tokenize.test.ts
git commit -m "feat: port tokenize (normalizeText, tokenCount, contentHash)"
```

---

## Task 3: Chunker

**Files:**
- Create: `src/chunker.ts`
- Test: `test/chunker.test.ts`, `test/splitter.test.ts`

- [ ] **Step 1: Write `test/chunker.test.ts`** (ports `test_chunker.jl`)

```typescript
import { test, expect } from "bun:test";
import { parseMarkdownSections, chunkDocument, splitToTokenBudget } from "../src/chunker.ts";
import { defaultChunkingConfig } from "../src/types.ts";

const md = `# Moves

Intro paragraph.

## Delve the Depths

**When you delve the depths**, roll +wits.

On a 10+, choose two.

## Secure an Advantage

**When you secure an advantage**, roll +heart.
`;

test("parseMarkdownSections", () => {
  const secs = parseMarkdownSections(md);
  expect(secs.length).toBe(3);
  expect(secs[0]!.headingPath).toEqual(["Moves"]);
  expect(secs[0]!.text).toContain("Intro paragraph");
  expect(secs[1]!.headingPath).toEqual(["Moves", "Delve the Depths"]);
  expect(secs[1]!.text.toLowerCase()).toContain("delve the depths");
  expect(secs[2]!.headingPath).toEqual(["Moves", "Secure an Advantage"]);
});

test("chunkDocument numbers chunks sequentially", () => {
  const md2 = `# Moves

Intro paragraph.

## Delve the Depths

**When you delve the depths**, roll +wits.

## Secure an Advantage

**When you secure an advantage**, roll +heart.
`;
  const cfg = defaultChunkingConfig({ minTokens: 1, maxTokens: 200, overlapTokens: 0 });
  const raws = chunkDocument(md2, cfg);
  expect(raws.length).toBe(3);
  expect(raws[0]!.headingPath).toEqual(["Moves"]);
  expect(raws[1]!.headingPath).toEqual(["Moves", "Delve the Depths"]);
  expect(raws[0]!.chunkOrder).toBe(0);
  expect(raws[1]!.chunkOrder).toBe(1);
  expect(raws[2]!.chunkOrder).toBe(2);
});

test("table kept atomic across token boundary", () => {
  const table = `| Weapon      | Damage | Weight |
|-------------|--------|--------|
| Iron Sword  | 3      | 1      |
| Bone Spear  | 4      | 2      |
| Blight Blade| 5      | 3      |`;
  const padding = Array(80).fill("word").join(" ");
  const text = padding + "\n\n" + table.trim();
  const pieces = splitToTokenBudget(text, { maxTokens: 100, overflow: "paragraph" });
  const withTable = pieces.filter((p) => p.includes("Iron Sword"));
  expect(withTable.length).toBe(1);
  expect(withTable[0]!).toContain("Bone Spear");
  expect(withTable[0]!).toContain("Blight Blade");
});

test("small table not split", () => {
  const table = `| A | B |
|---|---|
| 1 | 2 |
| 3 | 4 |`;
  const pieces = splitToTokenBudget(table.trim(), { maxTokens: 200 });
  expect(pieces.length).toBe(1);
  expect(pieces[0]!).toContain("| 3 | 4 |");
});

test("large table exceeding max kept whole", () => {
  const rows = Array.from({ length: 100 }, (_, i) =>
    `| item_${String(i + 1).padStart(3, "0")} | ${(i + 1) * 10} |`).join("\n");
  const table = "| Item | Value |\n|------|-------|\n" + rows;
  const pieces = splitToTokenBudget(table, { maxTokens: 50, overflow: "paragraph" });
  const first = pieces.filter((p) => p.includes("item_001"));
  expect(first.length).toBe(1);
  expect(first[0]!).toContain("item_100");
});
```

- [ ] **Step 2: Write `test/splitter.test.ts`** (ports `test_splitter.jl`)

```typescript
import { test, expect } from "bun:test";
import { splitToTokenBudget } from "../src/chunker.ts";

test("splitToTokenBudget", () => {
  expect(splitToTokenBudget("one two three", { maxTokens: 10, overlapTokens: 2, overflow: "paragraph" }))
    .toEqual(["one two three"]);

  const txt = "para one has several words here.\n\npara two has several words too.\n\npara three closes it out.";
  const parts2 = splitToTokenBudget(txt, { maxTokens: 8, overlapTokens: 0, overflow: "paragraph" });
  expect(parts2.length).toBeGreaterThanOrEqual(2);
  expect(parts2.every((p) => p.split(/\s+/).length <= 16)).toBe(true);

  const long = Array.from({ length: 50 }, (_, i) => `w${i + 1}`).join(" ");
  const parts3 = splitToTokenBudget(long, { maxTokens: 10, overlapTokens: 2, overflow: "token" });
  expect(parts3.length).toBeGreaterThanOrEqual(5);
  expect(parts3[1]!.split(/\s+/).length).toBeLessThanOrEqual(12);
});
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `bun test test/chunker.test.ts test/splitter.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 4: Write `src/chunker.ts`**

```typescript
import type { ChunkingConfig, Overflow, RawChunk, Section } from "./types.ts";
import { wsCount, wsTokens } from "./tokenize.ts";

export function parseMarkdownSections(md: string): Section[] {
  const sections: Section[] = [];
  const stack: string[] = [];
  let buf: string[] = [];
  let started = false;

  const flush = () => {
    if (started) {
      const t = buf.join("\n").trim();
      if (t !== "") sections.push({ headingPath: [...stack], text: t });
    }
    buf = [];
  };

  for (const line of md.split("\n")) {
    const m = line.match(/^(#{1,6})\s+(.+?)\s*$/);
    if (m) {
      flush();
      const level = m[1]!.length;
      const title = m[2]!;
      while (stack.length >= level) stack.pop();
      while (stack.length < level - 1) stack.push("");
      stack.push(title);
      started = true;
    } else {
      buf.push(line);
    }
  }
  flush();
  return sections;
}

function greedyPack(units: string[], maxTokens: number): string[] {
  const out: string[] = [];
  let cur: string[] = [];
  let curTokens = 0;
  for (const raw of units) {
    const u = raw.trim();
    if (u === "") continue;
    const uTokens = wsCount(u);
    if (curTokens + uTokens > maxTokens && curTokens > 0) {
      out.push(cur.join(" ").trim());
      cur = [];
      curTokens = 0;
    }
    if (uTokens > maxTokens) {
      if (curTokens > 0) {
        out.push(cur.join(" ").trim());
        cur = [];
        curTokens = 0;
      }
      out.push(u);
    } else {
      cur.push(u);
      curTokens += uTokens;
    }
  }
  if (curTokens > 0) out.push(cur.join(" ").trim());
  return out;
}

function applyOverlap(pieces: string[], overlapTokens: number): string[] {
  if (overlapTokens <= 0 || pieces.length <= 1) return pieces;
  const out: string[] = [pieces[0]!];
  for (let i = 1; i < pieces.length; i++) {
    const prev = wsTokens(pieces[i - 1]!);
    const tail = prev.slice(Math.max(prev.length - overlapTokens, 0));
    out.push(tail.join(" ") + " " + pieces[i]!);
  }
  return out;
}

export function splitToTokenBudget(
  text: string,
  opts: { maxTokens: number; overlapTokens?: number; overflow?: Overflow },
): string[] {
  const maxTokens = opts.maxTokens;
  const overlapTokens = opts.overlapTokens ?? 0;
  let overflow: Overflow = opts.overflow ?? "paragraph";

  if (wsCount(text) <= maxTokens) return [text];

  if (overflow === "paragraph") {
    const pieces = greedyPack(text.split(/\n\s*\n/), maxTokens);
    if (pieces.every((p) => wsCount(p) <= maxTokens)) return applyOverlap(pieces, overlapTokens);
    overflow = "sentence";
  }
  if (overflow === "sentence") {
    const pieces = greedyPack(text.split(/(?<=[.!?])\s+/), maxTokens);
    if (pieces.every((p) => wsCount(p) <= maxTokens)) return applyOverlap(pieces, overlapTokens);
    overflow = "token";
  }

  const docLines = text.split("\n");
  let curLines: string[] = [];
  let curTokens = 0;
  const pieces: string[] = [];
  let i = 0;
  while (i < docLines.length) {
    const line = docLines[i]!;
    if (line.trimStart().startsWith("|")) {
      const tblLines: string[] = [];
      let tblTokens = 0;
      while (i < docLines.length && docLines[i]!.trimStart().startsWith("|")) {
        tblLines.push(docLines[i]!);
        tblTokens += wsCount(docLines[i]!);
        i++;
      }
      if (curTokens + tblTokens > maxTokens && curTokens > 0) {
        pieces.push(curLines.join("\n").trim());
        curLines = [];
        curTokens = 0;
      }
      curLines.push(...tblLines);
      curTokens += tblTokens;
      continue;
    }
    const lineTokens = wsCount(line);
    if (curTokens + lineTokens > maxTokens && curTokens > 0) {
      pieces.push(curLines.join("\n").trim());
      curLines = [];
      curTokens = 0;
    }
    if (lineTokens > maxTokens) {
      const lineToks = wsTokens(line);
      let j = 0;
      while (j < lineToks.length) {
        const k = Math.min(j + maxTokens, lineToks.length);
        pieces.push(lineToks.slice(j, k).join(" "));
        j = k;
      }
      curLines = [];
      curTokens = 0;
    } else {
      curLines.push(line);
      curTokens += lineTokens;
    }
    i++;
  }
  if (curTokens > 0) pieces.push(curLines.join("\n").trim());
  return applyOverlap(pieces, overlapTokens);
}

export function chunkDocument(md: string, cfg: ChunkingConfig): RawChunk[] {
  const sections = parseMarkdownSections(md);
  const out: RawChunk[] = [];
  let pending: Section | null = null;

  const emit = (sec: Section) => {
    const pieces = splitToTokenBudget(sec.text, {
      maxTokens: cfg.maxTokens,
      overlapTokens: cfg.overlapTokens,
      overflow: cfg.overflow,
    });
    for (const p of pieces) {
      out.push({ headingPath: sec.headingPath, text: p, chunkOrder: out.length, page: "" });
    }
  };

  for (const sec of sections) {
    if (wsCount(sec.text) < cfg.minTokens) {
      pending = pending === null
        ? sec
        : { headingPath: pending.headingPath, text: pending.text + "\n\n" + sec.text };
      continue;
    }
    if (pending !== null) {
      emit({ headingPath: pending.headingPath, text: pending.text + "\n\n" + sec.text });
      pending = null;
    } else {
      emit(sec);
    }
  }
  if (pending !== null) emit(pending);
  return out;
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `bun test test/chunker.test.ts test/splitter.test.ts`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/chunker.ts test/chunker.test.ts test/splitter.test.ts
git commit -m "feat: port chunker (sections, token-budget splitter)"
```

---

## Task 4: Config

**Files:**
- Create: `src/config.ts`
- Test: `test/config.test.ts`

- [ ] **Step 1: Write `test/config.test.ts`**

```typescript
import { test, expect, afterEach } from "bun:test";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { loadConfig, resetConfigCache } from "../src/config.ts";

const origCwd = process.cwd();
afterEach(() => {
  process.chdir(origCwd);
  resetConfigCache();
});

test("loadConfig reads and validates ./tomerag.config.json", () => {
  const dir = mkdtempSync(join(tmpdir(), "tomerag-cfg-"));
  writeFileSync(join(dir, "tomerag.config.json"), JSON.stringify({ anthropicApiKey: "sk-ant-test" }));
  process.chdir(dir);
  resetConfigCache();
  expect(loadConfig().anthropicApiKey).toBe("sk-ant-test");
  rmSync(dir, { recursive: true, force: true });
});

test("loadConfig throws a helpful error when the file is missing", () => {
  const dir = mkdtempSync(join(tmpdir(), "tomerag-cfg-"));
  process.chdir(dir);
  resetConfigCache();
  expect(() => loadConfig()).toThrow(/tomerag\.config\.json/);
  rmSync(dir, { recursive: true, force: true });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bun test test/config.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `src/config.ts`**

```typescript
import { readFileSync } from "node:fs";
import { z } from "zod";

const ConfigSchema = z.object({
  anthropicApiKey: z.string().min(1).optional(),
});

export type TomeragConfig = z.infer<typeof ConfigSchema>;

const CONFIG_PATH = "./tomerag.config.json";
let cache: TomeragConfig | null = null;

export function loadConfig(): TomeragConfig {
  if (cache !== null) return cache;
  let raw: string;
  try {
    raw = readFileSync(CONFIG_PATH, "utf8");
  } catch {
    throw new Error(
      `TomeRAG config not found at ${CONFIG_PATH}. ` +
      `Create it with a JSON object like { "anthropicApiKey": "sk-ant-..." } ` +
      `(see tomerag.config.example.json).`,
    );
  }
  cache = ConfigSchema.parse(JSON.parse(raw));
  return cache;
}

/** Test-only: clear the module-local cache so a different cwd/file is re-read. */
export function resetConfigCache(): void {
  cache = null;
}

/** Resolve an Anthropic API key from explicit arg or config; throw if absent. */
export function requireAnthropicKey(explicit?: string): string {
  const key = explicit ?? loadConfig().anthropicApiKey;
  if (!key) {
    throw new Error(
      `Anthropic API key not set. Pass apiKey explicitly or add ` +
      `"anthropicApiKey" to ${CONFIG_PATH}.`,
    );
  }
  return key;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `bun test test/config.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/config.ts test/config.test.ts
git commit -m "feat: add zod-validated config loader"
```

---

## Task 5: Embedding backends

**Files:**
- Create: `src/backends/embedding.ts`
- Test: `test/embedding.test.ts`, `test/ollama.test.ts`

- [ ] **Step 1: Write `test/embedding.test.ts`** (ports the mock-embedding portion of `test_backends.jl`)

```typescript
import { test, expect } from "bun:test";
import { MockEmbeddingBackend } from "../src/backends/embedding.ts";

test("mock embedding backend", async () => {
  const b = new MockEmbeddingBackend({ dim: 4 });
  const v = await b.embed("hello world");
  expect(v.length).toBe(4);
  expect(v).toBeInstanceOf(Float32Array);

  expect(await b.embed("foo")).toEqual(await b.embed("foo"));
  expect(await b.embed("foo")).not.toEqual(await b.embed("bar"));

  const vs = await b.embed(["a", "b", "c"]);
  expect(vs.length).toBe(3);
  expect(vs.every((x) => x.length === 4)).toBe(true);

  // L2-normalized
  const norm = Math.sqrt(Array.from(v).reduce((s, x) => s + x * x, 0));
  expect(norm).toBeCloseTo(1, 5);
});

test("mock embedding default dim is 8", async () => {
  const b = new MockEmbeddingBackend();
  expect((await b.embed("x")).length).toBe(8);
});
```

- [ ] **Step 2: Write `test/ollama.test.ts`** (ports `test_ollama.jl`; live test gated)

```typescript
import { test, expect } from "bun:test";
import { OllamaBackend } from "../src/backends/embedding.ts";

test("ollama backend construction", () => {
  const b = new OllamaBackend({ model: "nomic-embed-text", baseUrl: "http://localhost:11434", dim: 768 });
  expect(b.model).toBe("nomic-embed-text");
  expect(b.dim).toBe(768);
  expect(b.baseUrl).toBe("http://localhost:11434");
  expect(b.batchSize).toBe(32);
});

test.skipIf(process.env["TOMERAG_LIVE_TESTS"] !== "1")("ollama live embed", async () => {
  const b = new OllamaBackend({ model: "nomic-embed-text", dim: 768 });
  const v = await b.embed("hello");
  expect(v.length).toBe(768);
  const vs = await b.embed(["alpha", "beta", "gamma"]);
  expect(vs.length).toBe(3);
  expect(vs.every((x) => x.length === 768)).toBe(true);
});
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `bun test test/embedding.test.ts test/ollama.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 4: Write `src/backends/embedding.ts`**

```typescript
import { createHash } from "node:crypto";
import { z } from "zod";

export abstract class EmbeddingBackend {
  abstract embed(text: string): Promise<Float32Array>;
  abstract embed(texts: string[]): Promise<Float32Array[]>;
}

export class MockEmbeddingBackend extends EmbeddingBackend {
  readonly dim: number;

  constructor(opts?: { dim?: number }) {
    super();
    this.dim = opts?.dim ?? 8;
  }

  override embed(text: string): Promise<Float32Array>;
  override embed(texts: string[]): Promise<Float32Array[]>;
  override async embed(input: string | string[]): Promise<Float32Array | Float32Array[]> {
    if (Array.isArray(input)) return input.map((t) => this.one(t));
    return this.one(input);
  }

  private one(text: string): Float32Array {
    const h = createHash("sha256").update(text, "utf8").digest(); // 32-byte Buffer
    const v = new Float32Array(this.dim);
    for (let i = 0; i < this.dim; i++) {
      const byte = h[i % h.length]!;
      v[i] = (byte / 255) * 2 - 1;
    }
    let n = 0;
    for (let i = 0; i < this.dim; i++) n += v[i]! * v[i]!;
    n = Math.sqrt(n);
    if (n !== 0) for (let i = 0; i < this.dim; i++) v[i] = v[i]! / n;
    return v;
  }
}

const OllamaResponse = z.object({ embeddings: z.array(z.array(z.number())) });

export class OllamaBackend extends EmbeddingBackend {
  readonly model: string;
  readonly dim: number;
  readonly baseUrl: string;
  readonly batchSize: number;

  constructor(opts: { model: string; dim: number; baseUrl?: string; batchSize?: number }) {
    super();
    this.model = opts.model;
    this.dim = opts.dim;
    this.baseUrl = opts.baseUrl ?? "http://localhost:11434";
    this.batchSize = opts.batchSize ?? 32;
  }

  private async call(input: string[]): Promise<Float32Array[]> {
    const url = this.baseUrl.replace(/\/+$/, "") + "/api/embed";
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: this.model, input }),
    });
    if (!resp.ok) {
      throw new Error(`Ollama returned HTTP ${resp.status}: ${await resp.text()}`);
    }
    const payload = OllamaResponse.parse(await resp.json());
    return payload.embeddings.map((e) => Float32Array.from(e));
  }

  override embed(text: string): Promise<Float32Array>;
  override embed(texts: string[]): Promise<Float32Array[]>;
  override async embed(input: string | string[]): Promise<Float32Array | Float32Array[]> {
    if (!Array.isArray(input)) return (await this.call([input]))[0]!;
    const out: Float32Array[] = [];
    for (let i = 0; i < input.length; i += this.batchSize) {
      out.push(...(await this.call(input.slice(i, i + this.batchSize))));
    }
    return out;
  }
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `bun test test/embedding.test.ts test/ollama.test.ts`
Expected: PASS (ollama live test reported as skipped).

- [ ] **Step 6: Commit**

```bash
git add src/backends/embedding.ts test/embedding.test.ts test/ollama.test.ts
git commit -m "feat: port embedding backends (mock, ollama)"
```

---

## Task 6: Classify backends — Mock + Heuristic

**Files:**
- Create: `src/backends/classify.ts`
- Test: `test/heuristic.test.ts`, `test/classify.test.ts`

- [ ] **Step 1: Write `test/heuristic.test.ts`** (ports `test_heuristic.jl`)

```typescript
import { test, expect } from "bun:test";
import { HeuristicBackend } from "../src/backends/classify.ts";

test("heuristic classify", async () => {
  const h = new HeuristicBackend();

  const out = await h.classify({
    text: "**When you delve the depths**, roll +wits. On a 10+...",
    headingPath: ["Moves", "Adventure Moves", "Delve the Depths"],
  });
  expect(out.contentType).toBe("move");
  expect(out.moveTrigger).not.toBeNull();
  expect(out.moveTrigger!.toLowerCase()).toContain("delve the depths");

  const out2 = await h.classify({
    text: "| Roll | Result |\n|---|---|\n| 1 | A |\n| 2 | B |",
    headingPath: ["Reference", "Random Events"],
  });
  expect(out2.contentType).toBe("table");

  const out3 = await h.classify({
    text: "NPC: Blight Walker\nHP 12, Armor 2\nAttack +3",
    headingPath: ["Bestiary", "Blight Walker"],
  });
  expect(out3.contentType).toBe("stat_block");
  expect(out3.npcName).toBe("Blight Walker");

  const out4 = await h.classify({
    text: "The corrupted forest stretches for miles...",
    headingPath: ["The World", "Geography"],
  });
  expect(out4.contentType).toBe("lore");
});
```

- [ ] **Step 2: Write `test/classify.test.ts`** (ports the mock/`classifyBatch`-default portions of `test_backends.jl`)

```typescript
import { test, expect } from "bun:test";
import { MockClassifyBackend, HeuristicBackend } from "../src/backends/classify.ts";
import type { RawChunk } from "../src/types.ts";

test("mock classify backend", async () => {
  const c = new MockClassifyBackend({ contentType: "mechanic", tags: ["x"] });
  const out = await c.classify({ text: "anything", headingPath: ["a"] });
  expect(out.contentType).toBe("mechanic");
  expect(out.tags).toEqual(["x"]);
  expect(out.moveTrigger).toBeNull();
});

test("classifyBatch default matches per-item classify", async () => {
  const b = new HeuristicBackend();
  const raws: RawChunk[] = [
    { headingPath: ["Moves", "Iron Vow"], text: "**When you swear upon iron**, roll +heart. On a 10+, your vow is strong.", chunkOrder: 1, page: "" },
    { headingPath: ["Bestiary", "Ironclad"], text: "HP 15, Armor 2, Attack: Blade 1d6.", chunkOrder: 2, page: "" },
    { headingPath: ["The World", "Geography"], text: "The Ironlands stretch far to the north, cold and unforgiving.", chunkOrder: 3, page: "" },
  ];
  const batch = await b.classifyBatch(raws);
  const single = await Promise.all(raws.map((r) => b.classify({ text: r.text, headingPath: r.headingPath })));
  expect(batch.length).toBe(3);
  for (let i = 0; i < 3; i++) {
    expect(batch[i]!.contentType).toBe(single[i]!.contentType);
    expect(batch[i]!.moveTrigger).toBe(single[i]!.moveTrigger);
  }
});
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `bun test test/heuristic.test.ts test/classify.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 4: Write `src/backends/classify.ts`** (Mock + Heuristic + abstract base; Claude added in Task 7)

```typescript
import type { Classification, ContentType, RawChunk } from "../types.ts";

export abstract class ClassifyBackend {
  abstract classify(input: { text: string; headingPath: string[] }): Promise<Classification>;

  async classifyBatch(raws: RawChunk[]): Promise<Classification[]> {
    const out: Classification[] = [];
    for (const r of raws) out.push(await this.classify({ text: r.text, headingPath: r.headingPath }));
    return out;
  }
}

export class MockClassifyBackend extends ClassifyBackend {
  private readonly contentType: ContentType;
  private readonly tags: string[];

  constructor(opts?: { contentType?: ContentType; tags?: string[] }) {
    super();
    this.contentType = opts?.contentType ?? "mechanic";
    this.tags = opts?.tags ?? [];
  }

  override async classify(): Promise<Classification> {
    return {
      contentType: this.contentType,
      tags: [...this.tags],
      moveTrigger: null,
      sceneType: null,
      encounterKey: null,
      npcName: null,
    };
  }
}

const MOVE_TRIGGER_PAT = /\*\*when\s+([^*]+?)\*\*/i;
const TABLE_PAT = /^\s*\|.*\|\s*$/gm;
const STAT_LINE_PAT = /\b(hp|hit points|armor|ac|attack)\b[^\n]*\d/i;
const TABLE_HEADING_PAT = /tables?|oracle|random/i;
const BESTIARY_HEADING_PAT = /bestiary|stat ?block|npcs?|monsters?/i;
const LORE_HEADING_PAT = /world|geography|history|cult|faction|lore/i;
const GM_HEADING_PAT = /running|gm ?(guide|advice)|how to run/i;

export class HeuristicBackend extends ClassifyBackend {
  override async classify(input: { text: string; headingPath: string[] }): Promise<Classification> {
    const { text, headingPath } = input;
    const joinedHeading = headingPath.join(" / ").toLowerCase();
    const base = { tags: [] as string[], moveTrigger: null, sceneType: null, encounterKey: null, npcName: null };

    const m = text.match(MOVE_TRIGGER_PAT);
    if (m) {
      return { ...base, contentType: "move", moveTrigger: m[1]!.trim() };
    }

    const tableMatches = text.match(TABLE_PAT);
    if ((tableMatches?.length ?? 0) >= 2 || TABLE_HEADING_PAT.test(joinedHeading)) {
      return { ...base, contentType: "table" };
    }

    if (BESTIARY_HEADING_PAT.test(joinedHeading) && STAT_LINE_PAT.test(text)) {
      const last = headingPath.length === 0 ? null : headingPath[headingPath.length - 1]!.trim();
      return { ...base, contentType: "stat_block", npcName: last && last !== "" ? last : null };
    }

    if (GM_HEADING_PAT.test(joinedHeading)) return { ...base, contentType: "gm_guidance" };
    if (LORE_HEADING_PAT.test(joinedHeading)) return { ...base, contentType: "lore" };
    return { ...base, contentType: "mechanic" };
  }
}
```

> **Porting note:** `TABLE_PAT` uses the `g` flag so `String.match` returns all matches (Julia `count(_TABLE_PAT, text)`). The other heading patterns are used only with `.test()` (no `g`), so no `lastIndex` state leaks.

- [ ] **Step 5: Run tests to verify they pass**

Run: `bun test test/heuristic.test.ts test/classify.test.ts`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/backends/classify.ts test/heuristic.test.ts test/classify.test.ts
git commit -m "feat: port classify backends (mock, heuristic) + batch default"
```

---

## Task 7: ClaudeBackend + forecastCost

**Files:**
- Modify: `src/backends/classify.ts` (append `ClaudeBackend`, `CostEstimate`, `forecastCost`, `extractJsonArray`)
- Test: `test/claude.test.ts`

- [ ] **Step 1: Write `test/claude.test.ts`** (ports the ClaudeBackend/forecast portions of `test_backends.jl`)

```typescript
import { test, expect } from "bun:test";
import { ClaudeBackend, forecastCost, type CostEstimate } from "../src/backends/classify.ts";
import type { RawChunk } from "../src/types.ts";

test("ClaudeBackend constructs with required fields", () => {
  const b = new ClaudeBackend({ apiKey: "sk-fake", contentTypes: new Set(["move", "mechanic", "lore"]) });
  expect(b.model).toBe("claude-haiku-4-5-20251001");
  expect(b.batchSize).toBe(20);
  expect(b.contentTypes.has("move")).toBe(true);
});

test("buildBatchPrompt structure", () => {
  const b = new ClaudeBackend({
    apiKey: "sk-fake",
    contentTypes: new Set(["move", "mechanic", "lore"]),
    systemHint: "PbtA",
  });
  const raws: RawChunk[] = [
    { headingPath: ["Chapter 3", "Iron Vow"], text: "**When you swear upon iron**, roll +heart.", chunkOrder: 1, page: "" },
    { headingPath: ["The World"], text: "The Ironlands are cold.", chunkOrder: 2, page: "" },
  ];
  const prompt = b.buildBatchPrompt(raws);
  expect(prompt).toContain("[1]");
  expect(prompt).toContain("[2]");
  expect(prompt).toContain("Iron Vow");
  expect(prompt).toContain("move");
  expect(prompt).toContain("mechanic");
  expect(prompt).toContain("content_type");
  expect(prompt).toContain("move_trigger");
});

test("extractJsonArray strips fences and commentary", () => {
  const b = new ClaudeBackend({ apiKey: "sk-fake", contentTypes: new Set(["mechanic"]) });
  const raw = 'Here you go:\n```json\n[{"content_type":"mechanic"}]\n```\nDone.';
  expect(b.extractJsonArray(raw)).toBe('[{"content_type":"mechanic"}]');
});

test("classifyBatch falls back to heuristic when the API call fails", async () => {
  // No network/key: the SDK call rejects, so the whole batch falls back to HeuristicBackend.
  const b = new ClaudeBackend({ apiKey: "sk-not-real", contentTypes: new Set(["move", "mechanic"]) });
  const raws: RawChunk[] = [
    { headingPath: ["Moves", "Iron Vow"], text: "**When you swear upon iron**, roll +heart.", chunkOrder: 0, page: "" },
  ];
  const res = await b.classifyBatch(raws);
  expect(res.length).toBe(1);
  expect(res[0]!.contentType).toBe("move");
  expect(res[0]!.moveTrigger!.toLowerCase()).toContain("swear upon iron");
});

test.skipIf(process.env["TOMERAG_LIVE_TESTS"] !== "1")("forecastCost live", async () => {
  const b = new ClaudeBackend({ contentTypes: new Set(["move", "mechanic", "lore"]), systemHint: "PbtA" });
  const raws: RawChunk[] = [
    { headingPath: ["Moves", "Iron Vow"], text: "**When you swear upon iron**, roll +heart.", chunkOrder: 1, page: "" },
    { headingPath: ["The World"], text: "The Ironlands are cold and ancient.", chunkOrder: 2, page: "" },
    { headingPath: ["Bestiary", "Troll"], text: "HP 30, Armor 1, Attack: Club 2d6.", chunkOrder: 3, page: "" },
  ];
  const est: CostEstimate = await forecastCost(b, raws);
  expect(est.model).toBe(b.model);
  expect(est.nChunks).toBe(3);
  expect(est.nBatches).toBe(1);
  expect(est.inputTokens).toBeGreaterThan(0);
  expect(est.outputTokens).toBe(3 * 50);
  expect(est.totalCostUsd).toBeGreaterThan(0);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bun test test/claude.test.ts`
Expected: FAIL — `ClaudeBackend`/`forecastCost` not exported.

- [ ] **Step 3: Append `ClaudeBackend` + cost estimation to `src/backends/classify.ts`**

Add these imports at the top of `src/backends/classify.ts`:

```typescript
import Anthropic from "@anthropic-ai/sdk";
import { requireAnthropicKey } from "../config.ts";
```

Append at the end of the file:

```typescript
export interface ClaudeBackendOptions {
  contentTypes: Set<ContentType>;
  apiKey?: string;
  model?: string;
  batchSize?: number;
  systemHint?: string;
}

export class ClaudeBackend extends ClassifyBackend {
  readonly contentTypes: Set<ContentType>;
  readonly model: string;
  readonly batchSize: number;
  readonly systemHint: string;
  private readonly client: Anthropic;
  private readonly fallback = new HeuristicBackend();

  constructor(opts: ClaudeBackendOptions) {
    super();
    this.contentTypes = opts.contentTypes;
    this.model = opts.model ?? "claude-haiku-4-5-20251001";
    this.batchSize = opts.batchSize ?? 20;
    this.systemHint = opts.systemHint ?? "";
    this.client = new Anthropic({ apiKey: requireAnthropicKey(opts.apiKey) });
  }

  buildSystemPrompt(): string {
    const hint = this.systemHint === "" ? "" : ` (${this.systemHint} system)`;
    return (
      `You are classifying chunks of text from an RPG rulebook${hint}. ` +
      `Respond with ONLY a JSON array — no markdown fences, no commentary, ` +
      `no text before or after the array.`
    );
  }

  buildBatchPrompt(batch: RawChunk[]): string {
    const n = batch.length;
    const typesStr = [...this.contentTypes].sort().join(", ");
    const lines: string[] = [];
    lines.push(`Classify these ${n} RPG rulebook chunks.`);
    lines.push(`Return a JSON array of EXACTLY ${n} objects (index 1 through ${n}), one per chunk, in input order.`);
    lines.push(`Do NOT split, merge, skip, or add extra entries. Exactly ${n} objects.`);
    lines.push("");
    lines.push(`Valid content_types: ${typesStr}`);
    lines.push("");
    lines.push("Each object has these fields:");
    lines.push("  content_type  : one of the valid types above (string)");
    lines.push("  tags          : array of keyword strings (may be empty)");
    lines.push('  move_trigger  : the trigger phrase if content_type is "move", else null');
    lines.push("  scene_type    : scene category if applicable, else null");
    lines.push("  encounter_key : encounter identifier if applicable, else null");
    lines.push('  npc_name      : NPC name if content_type is "stat_block", else null');
    lines.push("");
    batch.forEach((raw, i) => {
      const heading = raw.headingPath.length === 0 ? "(no heading)" : raw.headingPath.join(" / ");
      lines.push(`[${i + 1}] heading: ${heading}`);
      lines.push(`text: ${raw.text}`);
      lines.push("");
    });
    return lines.join("\n");
  }

  /** Extract a JSON array from a noisy LLM response via bracket counting. */
  extractJsonArray(text: string): string {
    const start = text.indexOf("[");
    if (start === -1) throw new Error("No JSON array found in response");
    let depth = 0;
    let inStr = false;
    let esc = false;
    for (let i = start; i < text.length; i++) {
      const c = text[i]!;
      if (esc) { esc = false; continue; }
      if (c === "\\") { esc = inStr; continue; }
      if (c === '"') { inStr = !inStr; continue; }
      if (inStr) continue;
      if (c === "[") depth++;
      else if (c === "]") { depth--; if (depth === 0) return text.slice(start, i + 1); }
    }
    throw new Error("Unterminated JSON array in response");
  }

  private parseClassification(item: Record<string, unknown>): Classification {
    const ct = item["content_type"];
    const sym = ct == null ? "mechanic" : String(ct);
    const contentType = (this.contentTypes.has(sym as ContentType) ? sym : "mechanic") as ContentType;

    const rawTags = item["tags"];
    const tags = Array.isArray(rawTags) ? rawTags.map((t) => String(t)) : [];

    const nullable = (v: unknown): string | null =>
      v == null || v === "null" ? null : String(v);

    return {
      contentType,
      tags,
      moveTrigger: nullable(item["move_trigger"]),
      sceneType: nullable(item["scene_type"]),
      encounterKey: nullable(item["encounter_key"]),
      npcName: nullable(item["npc_name"]),
    };
  }

  override async classify(input: { text: string; headingPath: string[] }): Promise<Classification> {
    const raw: RawChunk = { headingPath: input.headingPath, text: input.text, chunkOrder: 0, page: "" };
    return (await this.classifyBatch([raw]))[0]!;
  }

  override async classifyBatch(raws: RawChunk[]): Promise<Classification[]> {
    if (raws.length === 0) return [];
    const system = this.buildSystemPrompt();
    const results: Classification[] = new Array(raws.length);

    for (let start = 0; start < raws.length; start += this.batchSize) {
      const batch = raws.slice(start, start + this.batchSize);
      const prompt = this.buildBatchPrompt(batch);
      try {
        const resp = await this.client.messages.create({
          model: this.model,
          max_tokens: Math.max(4096, batch.length * 300),
          system,
          messages: [{ role: "user", content: prompt }],
        });
        const first = resp.content[0];
        const respText = first && first.type === "text" ? first.text : "";
        const parsed = JSON.parse(this.extractJsonArray(respText)) as Record<string, unknown>[];

        if (parsed.length < batch.length) {
          for (let i = 0; i < parsed.length; i++) {
            results[start + i] = this.parseClassification(parsed[i]!);
          }
          for (let i = parsed.length; i < batch.length; i++) {
            const r = batch[i]!;
            results[start + i] = await this.fallback.classify({ text: r.text, headingPath: r.headingPath });
          }
        } else {
          for (let i = 0; i < batch.length; i++) {
            results[start + i] = this.parseClassification(parsed[i]!);
          }
        }
      } catch {
        for (let i = 0; i < batch.length; i++) {
          const r = batch[i]!;
          results[start + i] = await this.fallback.classify({ text: r.text, headingPath: r.headingPath });
        }
      }
    }
    return results;
  }
}

export interface CostEstimate {
  model: string;
  nChunks: number;
  nBatches: number;
  inputTokens: number;
  outputTokens: number;
  totalCostUsd: number;
}

// Conservative fallback (Sonnet rates), USD per million tokens.
const FALLBACK_PRICING = { input: 3.0, output: 15.0 };
const LITELLM_URL =
  "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json";
let pricingCache: Record<string, unknown> | null = null;

async function fetchPricing(): Promise<Record<string, unknown>> {
  if (pricingCache !== null) return pricingCache;
  try {
    const resp = await fetch(LITELLM_URL, { signal: AbortSignal.timeout(10_000) });
    pricingCache = resp.ok ? ((await resp.json()) as Record<string, unknown>) : {};
  } catch {
    pricingCache = {};
  }
  return pricingCache;
}

async function modelPricing(model: string): Promise<{ input: number; output: number }> {
  const data = await fetchPricing();
  const entry = data[model] as Record<string, unknown> | undefined;
  if (entry && typeof entry["input_cost_per_token"] === "number" &&
      typeof entry["output_cost_per_token"] === "number") {
    return {
      input: (entry["input_cost_per_token"] as number) * 1_000_000,
      output: (entry["output_cost_per_token"] as number) * 1_000_000,
    };
  }
  return FALLBACK_PRICING;
}

/** Test-only: clear the per-process pricing cache. */
export function resetPricingCache(): void {
  pricingCache = null;
}

export async function forecastCost(
  backend: ClaudeBackend,
  raws: RawChunk[],
): Promise<CostEstimate> {
  if (raws.length === 0) {
    return { model: backend.model, nChunks: 0, nBatches: 0, inputTokens: 0, outputTokens: 0, totalCostUsd: 0 };
  }
  // ClaudeBackend keeps `client` private; reach the SDK through a fresh client
  // built from the same resolved key is not possible (key not exposed), so the
  // count_tokens call is issued via the backend's own helper below.
  const system = backend.buildSystemPrompt();
  let totalIn = 0;
  let nBatches = 0;
  for (let start = 0; start < raws.length; start += backend.batchSize) {
    const batch = raws.slice(start, start + backend.batchSize);
    const prompt = backend.buildBatchPrompt(batch);
    const tr = await backend.countTokens(system, prompt);
    totalIn += tr;
    nBatches += 1;
  }
  const estOutput = raws.length * 50;
  const pricing = await modelPricing(backend.model);
  const cost = (totalIn * pricing.input) / 1_000_000 + (estOutput * pricing.output) / 1_000_000;
  return {
    model: backend.model,
    nChunks: raws.length,
    nBatches,
    inputTokens: totalIn,
    outputTokens: estOutput,
    totalCostUsd: cost,
  };
}
```

Add this method to the `ClaudeBackend` class body (so `forecastCost` can reach the SDK without exposing the key):

```typescript
  async countTokens(system: string, prompt: string): Promise<number> {
    const tr = await this.client.messages.countTokens({
      model: this.model,
      system,
      messages: [{ role: "user", content: prompt }],
    });
    return tr.input_tokens;
  }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `bun test test/claude.test.ts`
Expected: PASS (`forecastCost live` skipped; the fallback test exercises the catch path with no network).

- [ ] **Step 5: Typecheck**

Run: `bun run typecheck`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/backends/classify.ts test/claude.test.ts
git commit -m "feat: port ClaudeBackend (batch API, heuristic fallback) + forecastCost"
```

---

## Task 8: Extraction backends

**Files:**
- Create: `src/backends/extraction.ts`
- Test: `test/extraction.test.ts`

- [ ] **Step 1: Write `test/extraction.test.ts`** (ports `test_extraction.jl`; Poppler/Vision live tests gated)

```typescript
import { test, expect } from "bun:test";
import { createHash } from "node:crypto";
import { mkdtempSync, writeFileSync, readFileSync, existsSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  MockExtractionBackend, CachingBackend, ExtractionBackend, splitPdftext,
} from "../src/backends/extraction.ts";
import type { PageText } from "../src/types.ts";

test("MockExtractionBackend", async () => {
  const pages: PageText[] = [
    { pageNum: 1, text: "# Iron Vow\nRoll +heart." },
    { pageNum: 2, text: "# Face Danger\nRoll +edge." },
  ];
  const b = new MockExtractionBackend(pages);
  const result = await b.extractPages("any/path.pdf");
  expect(result.length).toBe(2);
  expect(result[1]!.pageNum).toBe(2);

  const p = await b.extractPage("any/path.pdf", 2);
  expect(p.text).toBe("# Face Danger\nRoll +edge.");

  await expect(b.extractPage("any/path.pdf", 99)).rejects.toThrow();
});

test("splitPdftext splits on form-feed and skips blanks", () => {
  const p1 = splitPdftext("Page one content.\fPage two content.\f");
  expect(p1.length).toBe(2);
  expect(p1[0]).toEqual({ pageNum: 1, text: "Page one content." });
  expect(p1[1]).toEqual({ pageNum: 2, text: "Page two content." });

  const p2 = splitPdftext("Content\f\f\fMore content");
  expect(p2.length).toBe(2);
  expect(p2[0]).toEqual({ pageNum: 1, text: "Content" });
  expect(p2[1]).toEqual({ pageNum: 4, text: "More content" });
});

test("CachingBackend caches to disk and reuses", async () => {
  const dir = mkdtempSync(join(tmpdir(), "tomerag-cache-"));
  const pdf = join(dir, "test.pdf");
  writeFileSync(pdf, "fake pdf bytes for hashing");

  let calls = 0;
  class Counting extends ExtractionBackend {
    override async extractPages(): Promise<PageText[]> {
      calls++;
      return [{ pageNum: 1, text: "iron vow text" }, { pageNum: 2, text: "face danger text" }];
    }
  }
  const cache = new CachingBackend({ inner: new Counting(), cacheDir: join(dir, "cache") });

  const r1 = await cache.extractPages(pdf);
  expect(r1.length).toBe(2);
  expect(calls).toBe(1);

  const hash = createHash("sha256").update(readFileSync(pdf)).digest("hex");
  const cdir = join(dir, "cache", hash);
  expect(existsSync(join(cdir, "page_001.txt"))).toBe(true);
  expect(readFileSync(join(cdir, "page_001.txt"), "utf8")).toBe("iron vow text");

  await cache.extractPages(pdf); // second call: served from disk
  expect(calls).toBe(1);
});

test("CachingBackend keys by content, not filename", async () => {
  const dir = mkdtempSync(join(tmpdir(), "tomerag-cache-"));
  const a = join(dir, "a.pdf"); writeFileSync(a, "pdf content A");
  const b = join(dir, "b.pdf"); writeFileSync(b, "pdf content B");
  const cache = new CachingBackend({
    inner: new MockExtractionBackend([{ pageNum: 1, text: "content" }]),
    cacheDir: join(dir, "cache"),
  });
  await cache.extractPages(a);
  await cache.extractPages(b);
  const ha = createHash("sha256").update(readFileSync(a)).digest("hex");
  const hb = createHash("sha256").update(readFileSync(b)).digest("hex");
  expect(ha).not.toBe(hb);
  expect(readdirSync(join(dir, "cache")).sort()).toEqual([ha, hb].sort());
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bun test test/extraction.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `src/backends/extraction.ts`**

```typescript
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync,
} from "node:fs";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Anthropic from "@anthropic-ai/sdk";
import type { PageText } from "../types.ts";
import { requireAnthropicKey } from "../config.ts";

export abstract class ExtractionBackend {
  abstract extractPages(pdfPath: string): Promise<PageText[]>;

  async extractPage(pdfPath: string, pageNum: number): Promise<PageText> {
    const pages = await this.extractPages(pdfPath);
    const found = pages.find((p) => p.pageNum === pageNum);
    if (!found) throw new Error(`page ${pageNum} not found in ${pdfPath}`);
    return found;
  }
}

export class MockExtractionBackend extends ExtractionBackend {
  constructor(private readonly pages: PageText[]) {
    super();
  }
  override async extractPages(): Promise<PageText[]> {
    return this.pages;
  }
}

/** Split `pdftotext` output on form-feed; skip blank pages but keep 1-based original index. */
export function splitPdftext(output: string): PageText[] {
  const raw = output.split("\f");
  const result: PageText[] = [];
  raw.forEach((r, i) => {
    const text = r.trim();
    if (text !== "") result.push({ pageNum: i + 1, text });
  });
  return result;
}

function run(bin: string, args: string[]): string {
  const r = spawnSync(bin, args, { encoding: "utf8", maxBuffer: 256 * 1024 * 1024 });
  if (r.error) throw new Error(`${bin} not found or failed: ${r.error.message}`);
  if (r.status !== 0) throw new Error(`${bin} exited ${r.status}: ${r.stderr}`);
  return r.stdout;
}

export class PopplerBackend extends ExtractionBackend {
  override async extractPages(pdfPath: string): Promise<PageText[]> {
    return splitPdftext(run("pdftotext", ["-layout", pdfPath, "-"]));
  }

  override async extractPage(pdfPath: string, pageNum: number): Promise<PageText> {
    const out = run("pdftotext", ["-layout", "-f", String(pageNum), "-l", String(pageNum), pdfPath, "-"]).trim();
    if (out === "") throw new Error(`page ${pageNum} is blank or out of range in ${pdfPath}`);
    return { pageNum, text: out };
  }
}

export class CachingBackend extends ExtractionBackend {
  private readonly inner: ExtractionBackend;
  private readonly cacheDir: string;

  constructor(opts: { inner: ExtractionBackend; cacheDir: string }) {
    super();
    this.inner = opts.inner;
    this.cacheDir = opts.cacheDir;
  }

  override async extractPages(pdfPath: string): Promise<PageText[]> {
    const hash = createHash("sha256").update(readFileSync(pdfPath)).digest("hex");
    const dir = join(this.cacheDir, hash);

    if (existsSync(join(dir, "page_001.txt"))) {
      const cached: PageText[] = [];
      let i = 1;
      for (;;) {
        const f = join(dir, `page_${String(i).padStart(3, "0")}.txt`);
        if (!existsSync(f)) break;
        cached.push({ pageNum: i, text: readFileSync(f, "utf8") });
        i++;
      }
      if (cached.length > 0) return cached;
    }

    const results = await this.inner.extractPages(pdfPath);
    mkdirSync(dir, { recursive: true });
    for (const pt of results) {
      writeFileSync(join(dir, `page_${String(pt.pageNum).padStart(3, "0")}.txt`), pt.text);
    }
    return results;
  }
}

const VISION_PROMPT =
  "This is page {PAGE} of an RPG rulebook. Extract all text exactly as written. " +
  "Format as markdown: use # headings for chapter/section titles, ## for subsections, " +
  "**bold** for move names and keywords, > blockquotes for sidebars and boxed text, " +
  "and markdown tables for any tabular content. Preserve the reading order. " +
  "Output only the extracted text, nothing else.";

function pLimit(concurrency: number) {
  let active = 0;
  const queue: (() => void)[] = [];
  const next = () => {
    active--;
    const job = queue.shift();
    if (job) job();
  };
  return <T>(fn: () => Promise<T>): Promise<T> =>
    new Promise<T>((resolve, reject) => {
      const runJob = () => {
        active++;
        fn().then(resolve, reject).finally(next);
      };
      if (active < concurrency) runJob();
      else queue.push(runJob);
    });
}

export class VisionBackend extends ExtractionBackend {
  readonly model: string;
  readonly concurrency: number;
  readonly dpi: number;
  private readonly client: Anthropic;

  constructor(opts?: { apiKey?: string; model?: string; concurrency?: number; dpi?: number }) {
    super();
    this.model = opts?.model ?? "claude-haiku-4-5-20251001";
    this.concurrency = opts?.concurrency ?? 5;
    this.dpi = opts?.dpi ?? 150;
    this.client = new Anthropic({ apiKey: requireAnthropicKey(opts?.apiKey) });
  }

  private pageCount(pdfPath: string): number {
    const out = run("pdfinfo", [pdfPath]);
    const m = out.match(/Pages:\s+(\d+)/);
    if (!m) throw new Error(`could not determine page count for ${pdfPath}`);
    return parseInt(m[1]!, 10);
  }

  private renderPage(pdfPath: string, pageNum: number): Buffer {
    const tmp = mkdtempSync(join(tmpdir(), "tomerag-ppm-"));
    try {
      const prefix = join(tmp, "page");
      run("pdftoppm", [
        "-r", String(this.dpi), "-f", String(pageNum), "-l", String(pageNum),
        "-png", "-singlefile", pdfPath, prefix,
      ]);
      const png = prefix + ".png";
      if (!existsSync(png)) throw new Error(`pdftoppm did not produce ${png} for page ${pageNum}`);
      return readFileSync(png);
    } finally {
      rmSync(tmp, { recursive: true, force: true });
    }
  }

  override async extractPage(pdfPath: string, pageNum: number): Promise<PageText> {
    const png = this.renderPage(pdfPath, pageNum);
    const resp = await this.client.messages.create({
      model: this.model,
      max_tokens: 4096,
      messages: [{
        role: "user",
        content: [
          { type: "image", source: { type: "base64", media_type: "image/png", data: png.toString("base64") } },
          { type: "text", text: VISION_PROMPT.replace("{PAGE}", String(pageNum)) },
        ],
      }],
    });
    const first = resp.content[0];
    const text = first && first.type === "text" ? first.text : "";
    return { pageNum, text };
  }

  override async extractPages(pdfPath: string): Promise<PageText[]> {
    const n = this.pageCount(pdfPath);
    const limit = pLimit(this.concurrency);
    const pages = await Promise.all(
      Array.from({ length: n }, (_, i) => limit(() => this.extractPage(pdfPath, i + 1))),
    );
    return pages.sort((a, b) => a.pageNum - b.pageNum);
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `bun test test/extraction.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/backends/extraction.ts test/extraction.test.ts
git commit -m "feat: port extraction backends (mock, poppler, caching, vision)"
```

---

## Task 9: Storage

**Files:**
- Create: `src/storage.ts`
- Test: `test/storage.test.ts`

- [ ] **Step 1: Write `test/storage.test.ts`** (ports `test_storage.jl`)

```typescript
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bun test test/storage.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `src/storage.ts`**

```typescript
import { DuckDBInstance } from "@duckdb/node-api";
import type { DuckDBConnection } from "@duckdb/node-api";
import type { Chunk, ContentType, DocumentType, License, QueryResult, Source } from "./types.ts";

type FilterCol = "content_type" | "document_type" | "system" | "doc_id";
const ALLOWED_FILTERS: ReadonlySet<string> = new Set(["content_type", "document_type", "system", "doc_id"]);

async function connect(dbPath: string): Promise<DuckDBConnection> {
  const instance = await DuckDBInstance.create(dbPath);
  const conn = await instance.connect();
  await conn.run("SET hnsw_enable_experimental_persistence = true;");
  return conn;
}

function embeddingLiteral(emb: Float32Array, dim: number): string {
  return `[${Array.from(emb).join(",")}]::FLOAT[${dim}]`;
}

export async function initializeStore(src: Source): Promise<void> {
  const conn = await connect(src.dbPath);
  try {
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
  } finally {
    conn.closeSync();
  }
}

export async function insertChunks(src: Source, chunks: Chunk[]): Promise<number> {
  if (chunks.length === 0) return 0;
  const conn = await connect(src.dbPath);
  let inserted = 0;
  try {
    await conn.run("LOAD vss;");
    await conn.run("LOAD fts;");
    await conn.run("BEGIN;");
    for (const c of chunks) {
      const existing = await conn.runAndReadAll(
        "SELECT 1 FROM chunks WHERE doc_id = ? AND content_hash = ? LIMIT 1",
        [c.docId, c.contentHash],
      );
      if (existing.getRowObjectsJS().length > 0) continue;
      await conn.run(
        `INSERT INTO chunks VALUES
         (?, ?, ?, ?, ?, ${embeddingLiteral(c.embedding, src.embeddingDim)},
          ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        [
          c.id, c.sourceId, c.docId, c.docPath, c.text,
          c.embeddingModel, c.tokenCount, c.contentHash, c.documentType,
          c.system, c.edition, c.page, JSON.stringify(c.headingPath),
          c.chunkOrder, c.parentId, c.contentType, JSON.stringify(c.tags),
          c.moveTrigger, c.sceneType, c.encounterKey, c.npcName, c.license,
        ],
      );
      inserted++;
    }
    await conn.run("COMMIT;");
    if (inserted > 0) {
      await conn.run("PRAGMA drop_fts_index('chunks');");
      await conn.run("PRAGMA create_fts_index('chunks', 'id', 'text', stemmer='porter', overwrite=1);");
    }
  } catch (e) {
    try { await conn.run("ROLLBACK;"); } catch { /* ignore */ }
    throw e;
  } finally {
    conn.closeSync();
  }
  return inserted;
}

export async function sourceStats(
  src: Source,
): Promise<{ chunkCount: number; embeddingModel: string; embeddingDim: number }> {
  const conn = await connect(src.dbPath);
  try {
    const r = await conn.runAndReadAll("SELECT COUNT(*) AS n FROM chunks");
    const row = r.getRowObjectsJS()[0]!;
    return {
      chunkCount: Number(row["n"]),
      embeddingModel: src.embeddingModel,
      embeddingDim: src.embeddingDim,
    };
  } finally {
    conn.closeSync();
  }
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
  src: Source,
  queryEmbedding: Float32Array,
  opts?: { topK?: number; filters?: Partial<Record<FilterCol, string>> },
): Promise<QueryResult[]> {
  const topK = opts?.topK ?? 10;
  const { clause, values } = buildFilters(opts?.filters);
  const conn = await connect(src.dbPath);
  try {
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
  } finally {
    conn.closeSync();
  }
}

export async function bm25Search(
  src: Source,
  queryText: string,
  opts?: { topK?: number; filters?: Partial<Record<FilterCol, string>> },
): Promise<Array<{ chunk: Chunk; score: number }>> {
  const topK = opts?.topK ?? 10;
  const { clause, values } = buildFilters(opts?.filters);
  const conn = await connect(src.dbPath);
  try {
    await conn.run("LOAD fts;");
    const escaped = queryText.replace(/'/g, "''");
    const where = clause === ""
      ? "WHERE score IS NOT NULL"
      : clause.replace("WHERE ", "WHERE score IS NOT NULL AND ");
    const sql = `
      SELECT ${ROW_COLUMNS},
        fts_main_chunks.match_bm25(id, '${escaped}') AS score
      FROM chunks
      ${where}
      ORDER BY score DESC
      LIMIT ${topK}`;
    const r = await conn.runAndReadAll(sql, values);
    const rows = r.getRowObjectsJS() as Record<string, unknown>[];
    return rows.map((row) => ({ chunk: rowToChunk(row), score: Number(row["score"]) }));
  } finally {
    conn.closeSync();
  }
}
```

> **Porting notes:** (1) Embeddings and query vectors are SQL literals, never bind params (matches `scribe/`). (2) `score IS NOT NULL` references the output alias in `WHERE` — DuckDB supports this and the Julia reference relies on it. (3) `runAndReadAll` is used for the dedup `SELECT` because `conn.run` returns a result that may not expose rows directly; `getRowObjectsJS().length` is the existence check.

- [ ] **Step 4: Run test to verify it passes**

Run: `bun test test/storage.test.ts`
Expected: PASS. If `INSTALL vss`/`INSTALL fts` needs network on first run, ensure connectivity; the extensions are cached by DuckDB after first install.

- [ ] **Step 5: Commit**

```bash
git add src/storage.ts test/storage.test.ts
git commit -m "feat: port storage (init, insert+dedup, similarity, bm25)"
```

---

## Task 10: Ingest orchestration

**Files:**
- Create: `src/ingest.ts`
- Test: `test/ingest.test.ts`

- [ ] **Step 1: Write `test/ingest.test.ts`** (ports `test_ingest.jl`; `SourceRegistry` dropped — pass `Source` directly)

```typescript
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bun test test/ingest.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `src/ingest.ts`**

```typescript
import { randomUUID } from "node:crypto";
import { resolve, extname } from "node:path";
import type {
  Chunk, DocumentType, RawChunk, Source,
} from "./types.ts";
import { chunkDocument } from "./chunker.ts";
import { contentHash, tokenCount } from "./tokenize.ts";
import type { EmbeddingBackend } from "./backends/embedding.ts";
import type { ClassifyBackend } from "./backends/classify.ts";
import type { ExtractionBackend } from "./backends/extraction.ts";
import { insertChunks } from "./storage.ts";

export interface IngestOptions {
  source: Source;
  path: string;
  docId: string;
  documentType: DocumentType;
  format?: "auto" | "markdown" | "pdf";
  embedBackend: EmbeddingBackend;
  classifyBackend: ClassifyBackend;
  extractionBackend?: ExtractionBackend;
}

export function injectPageMarker(pageText: string, pageNum: number): string {
  const marker = `<!-- page ${pageNum} -->`;
  const nl = pageText.indexOf("\n");
  const line1 = nl < 0 ? pageText : pageText.slice(0, nl);
  const rest = nl < 0 ? "" : pageText.slice(nl + 1);
  if (line1.trim().startsWith("#")) {
    return `${line1}\n${marker}\n${rest}`;
  }
  return `${marker}\n${pageText}`;
}

export function assignPages(chunks: RawChunk[]): RawChunk[] {
  return chunks.map((rc) => {
    const matches = [...rc.text.matchAll(/<!-- page (\d+) -->/g)];
    const page = matches.length === 0 ? "" : matches[matches.length - 1]![1]!;
    const clean = rc.text.replace(/<!-- page \d+ -->\n?/g, "").trim();
    return { headingPath: rc.headingPath, text: clean, chunkOrder: rc.chunkOrder, page };
  });
}

function searchText(r: RawChunk): string {
  if (r.headingPath.length === 0) return r.text;
  return `${r.headingPath.join(" > ")}\n\n${r.text}`;
}

export async function ingest(opts: IngestOptions): Promise<number> {
  let format = opts.format ?? "auto";
  if (format === "auto") {
    format = extname(opts.path).toLowerCase() === ".pdf" ? "pdf" : "markdown";
  }
  if (format === "pdf" && !opts.extractionBackend) {
    throw new Error("extractionBackend is required for format=pdf");
  }

  let docText: string;
  if (format === "pdf") {
    const pages = await opts.extractionBackend!.extractPages(opts.path);
    docText = pages.map((p) => injectPageMarker(p.text, p.pageNum)).join("\n\n");
  } else {
    docText = await Bun.file(opts.path).text();
  }

  let raws = chunkDocument(docText, opts.source.chunking);
  if (raws.length === 0) return 0;
  if (format === "pdf") raws = assignPages(raws);

  const classified = await opts.classifyBackend.classifyBatch(raws);
  const searchTexts = raws.map(searchText);
  const embeddings = await opts.embedBackend.embed(searchTexts);
  const absPath = resolve(opts.path);

  const chunks: Chunk[] = raws.map((r, i) => {
    const cls = classified[i]!;
    return {
      id: randomUUID(),
      sourceId: opts.source.id,
      docId: opts.docId,
      docPath: absPath,
      text: searchTexts[i]!,
      embedding: embeddings[i]!,
      embeddingModel: opts.source.embeddingModel,
      tokenCount: tokenCount(r.text),
      contentHash: contentHash(r.text),
      documentType: opts.documentType,
      system: opts.source.system,
      edition: "",
      page: r.page,
      headingPath: r.headingPath,
      chunkOrder: r.chunkOrder,
      parentId: null,
      contentType: cls.contentType,
      tags: cls.tags,
      moveTrigger: cls.moveTrigger,
      sceneType: cls.sceneType,
      encounterKey: cls.encounterKey,
      npcName: cls.npcName,
      license: opts.source.license,
    };
  });

  return insertChunks(opts.source, chunks);
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `bun test test/ingest.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ingest.ts test/ingest.test.ts
git commit -m "feat: port ingest orchestration (markdown + pdf, page assignment)"
```

---

## Task 11: Public re-exports

**Files:**
- Create: `src/index.ts`
- Test: `test/index.test.ts`

- [ ] **Step 1: Write `test/index.test.ts`**

```typescript
import { test, expect } from "bun:test";
import * as TomeRAG from "../src/index.ts";

test("public surface is exported", () => {
  for (const name of [
    "defaultChunkingConfig", "DEFAULT_CONTENT_TYPES", "PBTA_CONTENT_TYPES",
    "YZE_CONTENT_TYPES", "normalizeText", "tokenCount", "contentHash",
    "parseMarkdownSections", "splitToTokenBudget", "chunkDocument", "loadConfig",
    "MockEmbeddingBackend", "OllamaBackend", "MockClassifyBackend",
    "HeuristicBackend", "ClaudeBackend", "forecastCost", "MockExtractionBackend",
    "PopplerBackend", "CachingBackend", "VisionBackend", "initializeStore",
    "insertChunks", "sourceStats", "similaritySearch", "bm25Search", "ingest",
  ]) {
    expect(typeof (TomeRAG as Record<string, unknown>)[name]).not.toBe("undefined");
  }
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bun test test/index.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `src/index.ts`**

```typescript
export type {
  License, DocumentType, ContentType, Overflow, ChunkingConfig, Chunk,
  QueryResult, Source, RawChunk, PageText, Section, Classification,
} from "./types.ts";
export { defaultChunkingConfig } from "./types.ts";
export { DEFAULT_CONTENT_TYPES, PBTA_CONTENT_TYPES, YZE_CONTENT_TYPES } from "./content-types.ts";
export { normalizeText, tokenCount, contentHash, wsTokens, wsCount } from "./tokenize.ts";
export { parseMarkdownSections, splitToTokenBudget, chunkDocument } from "./chunker.ts";
export { loadConfig, requireAnthropicKey, type TomeragConfig } from "./config.ts";
export { EmbeddingBackend, MockEmbeddingBackend, OllamaBackend } from "./backends/embedding.ts";
export {
  ClassifyBackend, MockClassifyBackend, HeuristicBackend, ClaudeBackend,
  forecastCost, type CostEstimate, type ClaudeBackendOptions,
} from "./backends/classify.ts";
export {
  ExtractionBackend, MockExtractionBackend, PopplerBackend, CachingBackend,
  VisionBackend, splitPdftext,
} from "./backends/extraction.ts";
export {
  initializeStore, insertChunks, sourceStats, similaritySearch, bm25Search, rowToChunk,
} from "./storage.ts";
export { ingest, injectPageMarker, assignPages, type IngestOptions } from "./ingest.ts";
```

- [ ] **Step 4: Run test + full suite + typecheck**

Run: `bun test test/index.test.ts && bun run typecheck`
Expected: PASS, no type errors.

- [ ] **Step 5: Commit**

```bash
git add src/index.ts test/index.test.ts
git commit -m "feat: add public re-export surface"
```

---

## Task 12: CLI

**Files:**
- Create: `src/cli/ingest.ts`
- Test: `test/cli.test.ts`

- [ ] **Step 1: Write `test/cli.test.ts`**

```typescript
import { test, expect } from "bun:test";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { mkdtempSync, writeFileSync } from "node:fs";

test("cli ingests a markdown doc with mock embed + heuristic classify", async () => {
  const dir = mkdtempSync(join(tmpdir(), "tomerag-cli-"));
  const db = join(dir, "t.duckdb");
  const mdPath = join(dir, "doc.md");
  writeFileSync(mdPath, "# Moves\n\n## Iron Vow\n\n**When you swear upon iron**, roll +heart. On a 10+, succeed.\n");
  const cfgPath = join(dir, "source.json");
  writeFileSync(cfgPath, JSON.stringify({
    id: "iron", name: "Ironsworn", system: "PbtA", dbPath: db,
    embeddingModel: "mock", embeddingDim: 8, license: "cc_by",
    chunking: { minTokens: 1, maxTokens: 200, overflow: "paragraph", overlapTokens: 0 },
    contentTypes: ["mechanic", "move", "lore"],
  }));

  const proc = Bun.spawnSync([
    "bun", "run", join(import.meta.dir, "..", "src", "cli", "ingest.ts"),
    "--source-config", cfgPath, "--doc-id", "iron-core",
    "--document-type", "core_rules", "--path", mdPath,
    "--classify", "heuristic", "--embed", "mock",
  ], { cwd: dir });

  const out = proc.stdout.toString();
  expect(proc.exitCode).toBe(0);
  expect(out).toMatch(/inserted/i);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bun test test/cli.test.ts`
Expected: FAIL — `src/cli/ingest.ts` does not exist (non-zero exit).

- [ ] **Step 3: Write `src/cli/ingest.ts`**

```typescript
import { parseArgs } from "node:util";
import { readFileSync } from "node:fs";
import { z } from "zod";
import type { ContentType, Source } from "../types.ts";
import { defaultChunkingConfig } from "../types.ts";
import { initializeStore } from "../storage.ts";
import { ingest } from "../ingest.ts";
import { MockEmbeddingBackend, OllamaBackend, type EmbeddingBackend } from "../backends/embedding.ts";
import { HeuristicBackend, ClaudeBackend, type ClassifyBackend } from "../backends/classify.ts";

const SourceConfigSchema = z.object({
  id: z.string(),
  name: z.string(),
  system: z.string(),
  dbPath: z.string(),
  embeddingModel: z.string(),
  embeddingDim: z.number().int().positive(),
  license: z.enum(["cc_by", "cc_by_sa", "ogl", "orc", "homebrew", "proprietary"]),
  chunking: z.object({
    minTokens: z.number().int(),
    maxTokens: z.number().int(),
    overflow: z.enum(["paragraph", "sentence", "token"]),
    overlapTokens: z.number().int(),
  }).partial().optional(),
  contentTypes: z.array(z.string()),
});

function loadSource(path: string): Source {
  const cfg = SourceConfigSchema.parse(JSON.parse(readFileSync(path, "utf8")));
  return {
    id: cfg.id,
    name: cfg.name,
    system: cfg.system,
    dbPath: cfg.dbPath,
    embeddingModel: cfg.embeddingModel,
    embeddingDim: cfg.embeddingDim,
    license: cfg.license,
    chunking: defaultChunkingConfig(cfg.chunking ?? {}),
    contentTypes: new Set(cfg.contentTypes as ContentType[]),
  };
}

const { values } = parseArgs({
  options: {
    "source-config": { type: "string" },
    "doc-id": { type: "string" },
    "document-type": { type: "string" },
    "path": { type: "string" },
    "classify": { type: "string", default: "heuristic" },
    "embed": { type: "string", default: "ollama" },
  },
});

for (const req of ["source-config", "doc-id", "document-type", "path"] as const) {
  if (!values[req]) {
    console.error(`Missing required --${req}`);
    process.exit(2);
  }
}

const source = loadSource(values["source-config"]!);

const embedBackend: EmbeddingBackend =
  values.embed === "mock"
    ? new MockEmbeddingBackend({ dim: source.embeddingDim })
    : new OllamaBackend({ model: source.embeddingModel, dim: source.embeddingDim });

const classifyBackend: ClassifyBackend =
  values.classify === "claude"
    ? new ClaudeBackend({ contentTypes: source.contentTypes, systemHint: source.system })
    : new HeuristicBackend();

await initializeStore(source);
const inserted = await ingest({
  source,
  path: values.path!,
  docId: values["doc-id"]!,
  documentType: values["document-type"] as Source["chunking"] extends never ? never : "core_rules" | "adventure" | "supplement" | "campaign",
  embedBackend,
  classifyBackend,
});

console.log(`source=${source.id} doc=${values["doc-id"]} → inserted ${inserted} chunk(s) into ${source.dbPath}`);
```

> **Porting note:** the `document-type` cast keeps it a `DocumentType`; if you prefer, replace the inline conditional type with `as DocumentType` after importing the type. Defaults: `--embed ollama`, `--classify heuristic` (spec §CLI).

- [ ] **Step 4: Run test to verify it passes**

Run: `bun test test/cli.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cli/ingest.ts test/cli.test.ts
git commit -m "feat: add CLI ingest runner"
```

---

## Task 13: Schema byte-compat integration test

**Files:**
- Create: `test/integration.test.ts`
- Keep: `test/fixtures/ironsworn_sample.md` (already present)

> **Design clarification (deviation from spec wording):** the spec says "dynamically import `scribe/src/rag/query.ts`". That module hard-requires a 768-dim Ollama embedding and resolves its DB path from env, so it cannot run against a mock-embedded fixture DB without Ollama. To prove the *frozen schema contract* without external services, the default test issues the **exact SQL shapes `query.ts` uses** (`array_cosine_similarity` + `fts_main_chunks.match_bm25`, selecting `id, text, heading_path, content_type, move_trigger, page`) directly against the produced DuckDB. A second, `TOMERAG_LIVE_TESTS`-gated test does the literal `query.ts` round-trip (requires Ollama + a 768-dim source). This satisfies the spec's intent (byte-compat with the scribe reader) while keeping the default suite hermetic.

- [ ] **Step 1: Write `test/integration.test.ts`**

```typescript
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

  // Re-run the exact column/SQL shapes scribe/src/rag/query.ts depends on.
  const instance = await DuckDBInstance.create(src.dbPath, { access_mode: "READ_ONLY" });
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
    expect(String(vrows[0]!["text"]).toLowerCase()).toContain("delve the depths");
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
    const { searchRules } = (await import(SCRIBE_QUERY)) as typeof import("/media/karim/Code-Drive/karimn-code/rpg-rules/plugins/ironsworn/scribe/src/rag/query.ts");
    const results = await searchRules("delve the depths", { k: 3 });
    expect(results.length).toBeGreaterThanOrEqual(1);
    expect(results[0]!).toHaveProperty("headingPath");
    expect(results[0]!).toHaveProperty("moveTrigger");
  },
);
```

- [ ] **Step 2: Run test to verify it fails, then passes once deps exist**

Run: `bun test test/integration.test.ts`
Expected: PASS for the default test (live test skipped). If it fails on `array_cosine_similarity`, confirm `INSTALL vss; LOAD vss;` ran in `initializeStore` (it does) and that the DB was created by this suite.

- [ ] **Step 3: Run the full suite + typecheck**

Run: `bun test && bun run typecheck`
Expected: all suites PASS (live tests skipped), zero type errors.

- [ ] **Step 4: Commit**

```bash
git add test/integration.test.ts
git commit -m "test: add schema byte-compat integration test"
```

---

## Task 14: README + final verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Rewrite `README.md`**

```markdown
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
```

- [ ] **Step 2: Final verification**

Run: `bun test && bun run typecheck && git status --porcelain`
Expected: all tests pass, no type errors, clean tree after the commit below.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README for Bun/TS port"
```

- [ ] **Step 4 (optional, DESTRUCTIVE — confirm with user): in-place directory rename**

Per the spec decision. Only do this if the user explicitly confirms; it changes the working directory and affects the second additional working directory. From the parent dir:
```bash
cd .. && mv tomerag-jl tomerag && cd tomerag
```
Expected: repo contents unchanged, `.git` intact, cwd is now `.../tomerag/tomerag`.

---

## Self-Review

**1. Spec coverage**

| Spec section | Covered by |
|---|---|
| Types (`Chunk`, `Source`, `ChunkingConfig`, `QueryResult`, `RawChunk`, `PageText`, content-type sets) | Task 1 |
| `normalizeText`/`tokenCount`/`contentHash` | Task 2 |
| Chunker (sections, ladder, atomic tables, min-token merge) | Task 3 |
| Config (zod, `./tomerag.config.json`, cached, no env fallback) | Task 4 |
| Embedding backends (Mock, Ollama) | Task 5 |
| Classify (Mock, Heuristic, default `classifyBatch`) | Task 6 |
| Claude backend (batch API, JSON extraction, heuristic fallback) + `forecastCost`/`CostEstimate` (count_tokens, LiteLLM pricing, Sonnet fallback) | Task 7 |
| Extraction (Mock, Poppler, Caching, Vision + inline `pLimit`) | Task 8 |
| Storage (`initializeStore`, `insertChunks` dedup+FTS rebuild, `sourceStats`, `similaritySearch`, `bm25Search`, `rowToChunk`) | Task 9 |
| Ingest (`ingest`, format resolve, marker injection, `assignPages`, heading-prefixed search text, UUID ids) | Task 10 |
| Public re-exports | Task 11 |
| CLI (`parseArgs`, source-config JSON, defaults) | Task 12 |
| Byte-compat integration test | Task 13 |
| Migration (rename, remove Julia, bun init, deps, gitignore, config example, README) | Task 0, Task 14 |

`SourceRegistry` intentionally dropped (spec Decisions: "Drop `SourceRegistry`"). REST server and `extract_structured.jl` intentionally not ported (spec Non-goals).

**2. Placeholder scan:** no TBD/TODO/"add error handling"/"similar to Task N". Every code step contains complete code; every run step has an exact command and expected result.

**3. Type consistency:** `Classification` (camelCase: `contentType`, `moveTrigger`, `sceneType`, `encounterKey`, `npcName`) is defined in Task 1 and consumed identically by `classify.ts` (Task 6/7) and `ingest.ts` (Task 10). `ClaudeBackend` exposes `buildSystemPrompt`, `buildBatchPrompt`, `extractJsonArray`, `countTokens`, `model`, `batchSize`, `contentTypes` — all used by `forecastCost` and the Task 7 tests with matching names. `RawChunk` carries `page: ""` for markdown everywhere it is constructed (chunker, ingest, tests). Storage `FilterCol` keys (`content_type`/`document_type`/`system`/`doc_id`) match the Julia allowed set and the test usages. `rowToChunk` is the single snake_case→camelCase boundary, exported and reused by both search helpers.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-18-bun-ts-port.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
