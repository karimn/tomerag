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
