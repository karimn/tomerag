import { homedir } from "node:os";
import { resolve } from "node:path";
import { similaritySearch, bm25Search, ollamaEmbed } from "@tomerag/client";
import type { ReadSource } from "@tomerag/client";

const OLLAMA_BASE_URL = process.env["OLLAMA_BASE_URL"] ?? "http://localhost:11434";

function resolveDbPath(): string {
  const explicit = process.env["TROPHY_GOLD_DB_PATH"];
  if (explicit) return resolve(explicit);

  const home = homedir();
  return resolve(home, ".rpg-data", "trophy-gold", "trophy-gold.duckdb");
}

let _source: ReadSource | null = null;

function getSource(): ReadSource {
  if (_source === null) {
    _source = {
      id: "trophy-gold",
      system: "Trophy Gold",
      dbPath: resolveDbPath(),
      embeddingModel: "nomic-embed-text",
      embeddingDim: 768,
    };
  }
  return _source;
}

export function resetSourceCache(): void {
  _source = null;
}

export interface SearchOptions {
  topK?: number;
  contentType?: string;
}

export interface ChunkResult {
  id: string;
  text: string;
  headingPath: string[];
  contentType: string;
  page: string;
  score: number;
}

const RRF_K = 60;

export async function searchRules(
  query: string,
  opts?: SearchOptions,
): Promise<ChunkResult[]> {
  const source = getSource();
  const topK = opts?.topK ?? 5;

  const [queryEmb] = await ollamaEmbed([query], {
    model: source.embeddingModel,
    baseUrl: OLLAMA_BASE_URL,
  });

  const filters = opts?.contentType
    ? { content_type: opts.contentType }
    : undefined;

  const [vectorResults, bm25Results] = await Promise.all([
    similaritySearch(source, queryEmb, { topK: topK * 3, filters }),
    bm25Search(source, query, { topK: topK * 3, filters }),
  ]);

  const scores = new Map<string, number>();
  const map = new Map<string, ChunkResult>();

  for (const r of vectorResults) {
    const id = r.chunk.id;
    scores.set(id, (scores.get(id) ?? 0) + 1 / (RRF_K + r.rank));
    if (!map.has(id)) {
      map.set(id, {
        id,
        text: r.chunk.text,
        headingPath: r.chunk.headingPath,
        contentType: r.chunk.contentType,
        page: r.chunk.page,
        score: 0,
      });
    }
  }

  for (let i = 0; i < bm25Results.length; i++) {
    const r = bm25Results[i]!;
    const id = r.chunk.id;
    scores.set(id, (scores.get(id) ?? 0) + 1 / (RRF_K + i + 1));
    if (!map.has(id)) {
      map.set(id, {
        id,
        text: r.chunk.text,
        headingPath: r.chunk.headingPath,
        contentType: r.chunk.contentType,
        page: r.chunk.page,
        score: 0,
      });
    }
  }

  return [...scores.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, topK)
    .map(([id, score]) => ({ ...map.get(id)!, score }));
}
