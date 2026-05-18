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
  atomicPatterns: RegExp[];
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
