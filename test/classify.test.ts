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
