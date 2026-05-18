import { test, expect } from "bun:test";
import { MockEmbeddingBackend } from "../src/backends/embedding.ts";

test("mock embedding backend", async () => {
  const b = new MockEmbeddingBackend({ dim: 4 });
  const v = await b.embed("hello world");
  expect(v.length).toBe(4);
  expect(v).toBeInstanceOf(Float32Array);

  expect(await b.embed("foo")).toEqual(await b.embed("foo"));
  expect(await b.embed("foo")).not.toEqual(await b.embed("bar"));

  const vs = await b.embed(["a", "b", "c"]);
  expect(vs.length).toBe(3);
  expect(vs.every((x) => x.length === 4)).toBe(true);

  // L2-normalized
  const norm = Math.sqrt(Array.from(v).reduce((s, x) => s + x * x, 0));
  expect(norm).toBeCloseTo(1, 5);
});

test("mock embedding default dim is 8", async () => {
  const b = new MockEmbeddingBackend();
  expect((await b.embed("x")).length).toBe(8);
});
