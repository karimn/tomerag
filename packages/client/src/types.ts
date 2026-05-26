export type License =
  | "cc_by" | "cc_by_sa" | "ogl" | "orc" | "homebrew" | "proprietary";

export type DocumentType =
  | "core_rules" | "adventure" | "supplement" | "campaign";

export type ContentType =
  | "mechanic" | "lore" | "adventure_scene" | "table" | "stat_block"
  | "example" | "gm_guidance" | "flavor" | "procedure" | "boxed_text"
  | "move" | "gm_move" | "playbook" | "oracle" | "front"
  | "faction" | "gear";

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

export interface ReadSource {
  id: string;
  system: string;
  dbPath: string;
  embeddingModel: string;
  embeddingDim: number;
}
