import { test, expect } from "bun:test";
import { splitToTokenBudget } from "../src/chunker.ts";

test("splitToTokenBudget", () => {
  expect(splitToTokenBudget("one two three", { maxTokens: 10, overlapTokens: 2, overflow: "paragraph" }))
    .toEqual(["one two three"]);

  const txt = "para one has several words here.\n\npara two has several words too.\n\npara three closes it out.";
  const parts2 = splitToTokenBudget(txt, { maxTokens: 8, overlapTokens: 0, overflow: "paragraph" });
  expect(parts2.length).toBeGreaterThanOrEqual(2);
  expect(parts2.every((p) => p.split(/\s+/).length <= 16)).toBe(true);

  const long = Array.from({ length: 50 }, (_, i) => `w${i + 1}`).join(" ");
  const parts3 = splitToTokenBudget(long, { maxTokens: 10, overlapTokens: 2, overflow: "token" });
  expect(parts3.length).toBeGreaterThanOrEqual(5);
  expect(parts3[1]!.split(/\s+/).length).toBeLessThanOrEqual(12);
});
