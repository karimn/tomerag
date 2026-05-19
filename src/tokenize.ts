import { createHash } from "node:crypto";

/** Whitespace token split. Returns [] for empty/blank input (matches Julia `split("")`). */
export function wsTokens(s: string): string[] {
  const t = s.trim();
  return t === "" ? [] : t.split(/\s+/);
}

function wsCount(s: string): number {
  return wsTokens(s).length;
}

/** Lowercase, trim, collapse internal whitespace. Used only for dedup hashing. */
export function normalizeText(s: string): string {
  return s.toLowerCase().trim().replace(/\s+/g, " ");
}

/** Whitespace token count. Fast approximation adequate for chunk budgeting. */
export function tokenCount(s: string): number {
  return wsCount(s);
}

/** SHA-256 of the normalized text, lowercase hex. */
export function contentHash(s: string): string {
  return createHash("sha256").update(normalizeText(s), "utf8").digest("hex");
}
