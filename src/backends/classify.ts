import type { Classification, ContentType, RawChunk } from "../types.ts";
import Anthropic from "@anthropic-ai/sdk";
import { requireAnthropicKey } from "../config.ts";

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
    const results: Classification[] = [];
    for (let i = 0; i < raws.length; i++) results.push(undefined as unknown as Classification);

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

  async countTokens(system: string, prompt: string): Promise<number> {
    const tr = await this.client.messages.countTokens({
      model: this.model,
      system,
      messages: [{ role: "user", content: prompt }],
    });
    return tr.input_tokens;
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
