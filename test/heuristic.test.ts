import { test, expect } from "bun:test";
import { HeuristicBackend } from "../src/backends/classify.ts";

test("heuristic classify", async () => {
  const h = new HeuristicBackend();

  const out = await h.classify({
    text: "**When you delve the depths**, roll +wits. On a 10+...",
    headingPath: ["Moves", "Adventure Moves", "Delve the Depths"],
  });
  expect(out.contentType).toBe("move");
  expect(out.moveTrigger).not.toBeNull();
  expect(out.moveTrigger!.toLowerCase()).toContain("delve the depths");

  const out2 = await h.classify({
    text: "| Roll | Result |\n|---|---|\n| 1 | A |\n| 2 | B |",
    headingPath: ["Reference", "Random Events"],
  });
  expect(out2.contentType).toBe("table");

  const out3 = await h.classify({
    text: "NPC: Blight Walker\nHP 12, Armor 2\nAttack +3",
    headingPath: ["Bestiary", "Blight Walker"],
  });
  expect(out3.contentType).toBe("stat_block");
  expect(out3.npcName).toBe("Blight Walker");

  const out4 = await h.classify({
    text: "The corrupted forest stretches for miles...",
    headingPath: ["The World", "Geography"],
  });
  expect(out4.contentType).toBe("lore");
});
