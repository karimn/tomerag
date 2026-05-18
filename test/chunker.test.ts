import { test, expect } from "bun:test";
import { parseMarkdownSections, chunkDocument, splitToTokenBudget } from "../src/chunker.ts";
import { defaultChunkingConfig } from "../src/types.ts";

const md = `# Moves

Intro paragraph.

## Delve the Depths

**When you delve the depths**, roll +wits.

On a 10+, choose two.

## Secure an Advantage

**When you secure an advantage**, roll +heart.
`;

test("parseMarkdownSections", () => {
  const secs = parseMarkdownSections(md);
  expect(secs.length).toBe(3);
  expect(secs[0]!.headingPath).toEqual(["Moves"]);
  expect(secs[0]!.text).toContain("Intro paragraph");
  expect(secs[1]!.headingPath).toEqual(["Moves", "Delve the Depths"]);
  expect(secs[1]!.text.toLowerCase()).toContain("delve the depths");
  expect(secs[2]!.headingPath).toEqual(["Moves", "Secure an Advantage"]);
});

test("chunkDocument numbers chunks sequentially", () => {
  const md2 = `# Moves

Intro paragraph.

## Delve the Depths

**When you delve the depths**, roll +wits.

## Secure an Advantage

**When you secure an advantage**, roll +heart.
`;
  const cfg = defaultChunkingConfig({ minTokens: 1, maxTokens: 200, overlapTokens: 0 });
  const raws = chunkDocument(md2, cfg);
  expect(raws.length).toBe(3);
  expect(raws[0]!.headingPath).toEqual(["Moves"]);
  expect(raws[1]!.headingPath).toEqual(["Moves", "Delve the Depths"]);
  expect(raws[0]!.chunkOrder).toBe(0);
  expect(raws[1]!.chunkOrder).toBe(1);
  expect(raws[2]!.chunkOrder).toBe(2);
});

test("table kept atomic across token boundary", () => {
  const table = `| Weapon      | Damage | Weight |
|-------------|--------|--------|
| Iron Sword  | 3      | 1      |
| Bone Spear  | 4      | 2      |
| Blight Blade| 5      | 3      |`;
  const padding = Array(80).fill("word").join(" ");
  const text = padding + "\n\n" + table.trim();
  const pieces = splitToTokenBudget(text, { maxTokens: 100, overflow: "paragraph" });
  const withTable = pieces.filter((p) => p.includes("Iron Sword"));
  expect(withTable.length).toBe(1);
  expect(withTable[0]!).toContain("Bone Spear");
  expect(withTable[0]!).toContain("Blight Blade");
});

test("small table not split", () => {
  const table = `| A | B |
|---|---|
| 1 | 2 |
| 3 | 4 |`;
  const pieces = splitToTokenBudget(table.trim(), { maxTokens: 200 });
  expect(pieces.length).toBe(1);
  expect(pieces[0]!).toContain("| 3 | 4 |");
});

test("large table exceeding max kept whole", () => {
  const rows = Array.from({ length: 100 }, (_, i) =>
    `| item_${String(i + 1).padStart(3, "0")} | ${(i + 1) * 10} |`).join("\n");
  const table = "| Item | Value |\n|------|-------|\n" + rows;
  const pieces = splitToTokenBudget(table, { maxTokens: 50, overflow: "paragraph" });
  const first = pieces.filter((p) => p.includes("item_001"));
  expect(first.length).toBe(1);
  expect(first[0]!).toContain("item_100");
});
