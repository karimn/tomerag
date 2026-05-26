import type { ContentType, DocumentType, License, ReadSource } from "@tomerag/client";

export type { ContentType, DocumentType, License } from "@tomerag/client";

export type Overflow = "paragraph" | "sentence" | "token";

export interface ChunkingConfig {
  minTokens: number;
  maxTokens: number;
  overflow: Overflow;
  overlapTokens: number;
  atomicPatterns: RegExp[]; // defined for parity with TomeRAG.jl; the chunker does not read this field (table atomicity is hardcoded)
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

export interface Source extends ReadSource {
  name: string;
  license: License;
  chunking: ChunkingConfig;
  contentTypes: Set<ContentType>;
}

export interface RawChunk {
  headingPath: string[];
  text: string;
  chunkOrder: number;
  page: string;
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
