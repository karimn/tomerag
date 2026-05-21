import { test, expect, afterEach } from "bun:test";
import { NomicHostedBackend } from "../src/backends/embedding.ts";

const origFetch = globalThis.fetch;
const origKey = process.env["NOMIC_API_KEY"];

afterEach(() => {
  globalThis.fetch = origFetch;
  if (origKey === undefined) delete process.env["NOMIC_API_KEY"];
  else process.env["NOMIC_API_KEY"] = origKey;
});

test("nomic backend construction with explicit apiKey", () => {
  const b = new NomicHostedBackend({ apiKey: "nk-test", model: "nomic-embed-text-v1.5", dim: 768 });
  expect(b.model).toBe("nomic-embed-text-v1.5");
  expect(b.dim).toBe(768);
  expect(b.batchSize).toBe(32);
});

test("nomic backend defaults model and dim", () => {
  const b = new NomicHostedBackend({ apiKey: "nk-test" });
  expect(b.model).toBe(NomicHostedBackend.DEFAULT_MODEL);
  expect(b.dim).toBe(NomicHostedBackend.DEFAULT_DIM);
});

test("nomic backend picks up NOMIC_API_KEY env var", () => {
  process.env["NOMIC_API_KEY"] = "nk-from-env";
  expect(() => new NomicHostedBackend()).not.toThrow();
});

test("nomic backend throws a helpful error when no key is available", () => {
  delete process.env["NOMIC_API_KEY"];
  expect(() => new NomicHostedBackend()).toThrow(/NOMIC_API_KEY/);
});

test("nomic embed sends Bearer auth + {model, texts} body", async () => {
  process.env["NOMIC_API_KEY"] = "nk-test";
  let captured: { url: string; init: RequestInit } | null = null;
  globalThis.fetch = (async (url: string, init: RequestInit) => {
    captured = { url, init };
    return new Response(JSON.stringify({ embeddings: [[0.1, 0.2, 0.3]] }), { status: 200 });
  }) as unknown as typeof fetch;

  const b = new NomicHostedBackend({ dim: 3 });
  const v = await b.embed("hello");
  expect(v.length).toBe(3);
  expect(captured).not.toBeNull();
  expect(captured!.url).toBe(NomicHostedBackend.ENDPOINT);
  const headers = captured!.init.headers as Record<string, string>;
  expect(headers["Authorization"]).toBe("Bearer nk-test");
  const body = JSON.parse(captured!.init.body as string);
  expect(body.model).toBe(NomicHostedBackend.DEFAULT_MODEL);
  expect(body.texts).toEqual(["hello"]);
});

test("nomic embed batches arrays at batchSize", async () => {
  process.env["NOMIC_API_KEY"] = "nk-test";
  const calls: string[][] = [];
  globalThis.fetch = (async (_url: string, init: RequestInit) => {
    const body = JSON.parse(init.body as string);
    calls.push(body.texts);
    return new Response(
      JSON.stringify({ embeddings: body.texts.map(() => [0, 0]) }),
      { status: 200 },
    );
  }) as unknown as typeof fetch;

  const b = new NomicHostedBackend({ dim: 2, batchSize: 2 });
  const out = await b.embed(["a", "b", "c", "d", "e"]);
  expect(out.length).toBe(5);
  expect(calls.map((c) => c.length)).toEqual([2, 2, 1]);
});

test("nomic embed surfaces HTTP errors", async () => {
  process.env["NOMIC_API_KEY"] = "nk-test";
  globalThis.fetch = (async () =>
    new Response("invalid api key", { status: 401 })) as unknown as typeof fetch;

  const b = new NomicHostedBackend({ dim: 2 });
  await expect(b.embed("hello")).rejects.toThrow(/Nomic API returned HTTP 401/);
});

test.skipIf(process.env["TOMERAG_LIVE_TESTS"] !== "1")("nomic live embed", async () => {
  const b = new NomicHostedBackend();
  const v = await b.embed("hello");
  expect(v.length).toBe(NomicHostedBackend.DEFAULT_DIM);
  const vs = await b.embed(["alpha", "beta", "gamma"]);
  expect(vs.length).toBe(3);
  expect(vs.every((x) => x.length === NomicHostedBackend.DEFAULT_DIM)).toBe(true);
});
