import { parseArgs } from "node:util";
import { readFileSync } from "node:fs";
import { z } from "zod";
import type { ContentType, DocumentType, Source } from "../types.ts";
import { defaultChunkingConfig } from "../types.ts";
import { initializeStore } from "../storage.ts";
import { ingest } from "../ingest.ts";
import { MockEmbeddingBackend, NomicHostedBackend, OllamaBackend } from "../backends/embedding.ts";
import type { EmbeddingBackend } from "../backends/embedding.ts";
import { HeuristicBackend, ClaudeBackend } from "../backends/classify.ts";
import type { ClassifyBackend } from "../backends/classify.ts";

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
    "nomic-api-key": { type: "string" },
  },
});

for (const req of ["source-config", "doc-id", "document-type", "path"] as const) {
  if (!values[req]) {
    console.error(`Missing required --${req}`);
    process.exit(2);
  }
}

const source = loadSource(values["source-config"]!);

function buildEmbedBackend(): EmbeddingBackend {
  switch (values.embed) {
    case "mock":
      return new MockEmbeddingBackend({ dim: source.embeddingDim });
    case "ollama":
      return new OllamaBackend({ model: source.embeddingModel, dim: source.embeddingDim });
    case "nomic":
      return new NomicHostedBackend({
        apiKey: values["nomic-api-key"],
        model: source.embeddingModel,
        dim: source.embeddingDim,
      });
    default:
      console.error(`Unknown --embed value: ${values.embed} (expected mock | ollama | nomic)`);
      process.exit(2);
  }
}

const embedBackend: EmbeddingBackend = buildEmbedBackend();

const classifyBackend: ClassifyBackend =
  values.classify === "claude"
    ? new ClaudeBackend({ contentTypes: source.contentTypes, systemHint: source.system })
    : new HeuristicBackend();

await initializeStore(source);
const inserted = await ingest({
  source,
  path: values.path!,
  docId: values["doc-id"]!,
  documentType: values["document-type"] as DocumentType,
  embedBackend,
  classifyBackend,
});

console.log(`source=${source.id} doc=${values["doc-id"]} → inserted ${inserted} chunk(s) into ${source.dbPath}`);
