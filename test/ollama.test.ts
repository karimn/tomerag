import { test, expect } from "bun:test";
import { OllamaBackend } from "../src/backends/embedding.ts";

test("ollama backend construction", () => {
  const b = new OllamaBackend({ model: "nomic-embed-text", baseUrl: "http://localhost:11434", dim: 768 });
  expect(b.model).toBe("nomic-embed-text");
  expect(b.dim).toBe(768);
  expect(b.baseUrl).toBe("http://localhost:11434");
  expect(b.batchSize).toBe(32);
});

test.skipIf(process.env["TOMERAG_LIVE_TESTS"] !== "1")("ollama live embed", async () => {
  const b = new OllamaBackend({ model: "nomic-embed-text", dim: 768 });
  const v = await b.embed("hello");
  expect(v.length).toBe(768);
  const vs = await b.embed(["alpha", "beta", "gamma"]);
  expect(vs.length).toBe(3);
  expect(vs.every((x) => x.length === 768)).toBe(true);
});
