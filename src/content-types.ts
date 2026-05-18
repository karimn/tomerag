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
