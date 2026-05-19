import { test, expect } from "bun:test";
import { tokenCount, normalizeText, contentHash } from "../src/tokenize.ts";

test("tokenize", () => {
  expect(tokenCount("hello world")).toBe(2);
  expect(tokenCount("  hello   world \n foo ")).toBe(3);
  expect(tokenCount("")).toBe(0);

  expect(normalizeText(" Hello\tWorld \n")).toBe("hello world");

  const h1 = contentHash("Hello World");
  const h2 = contentHash("hello world");
  const h3 = contentHash("hello  world");
  expect(h1).toBe(h2);
  expect(h2).toBe(h3);
  expect(h1.length).toBe(64);
});
