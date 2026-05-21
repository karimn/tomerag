import { createHash } from "node:crypto";
import { z } from "zod";
import { requireNomicKey } from "../config.ts";

export abstract class EmbeddingBackend {
  abstract embed(text: string): Promise<Float32Array>;
  abstract embed(texts: string[]): Promise<Float32Array[]>;
}

export class MockEmbeddingBackend extends EmbeddingBackend {
  readonly dim: number;

  constructor(opts?: { dim?: number }) {
    super();
    this.dim = opts?.dim ?? 8;
  }

  override embed(text: string): Promise<Float32Array>;
  override embed(texts: string[]): Promise<Float32Array[]>;
  override async embed(input: string | string[]): Promise<Float32Array | Float32Array[]> {
    if (Array.isArray(input)) return input.map((t) => this.one(t));
    return this.one(input);
  }

  private one(text: string): Float32Array {
    const h = createHash("sha256").update(text, "utf8").digest(); // 32-byte Buffer
    const v = new Float32Array(this.dim);
    for (let i = 0; i < this.dim; i++) {
      const byte = h[i % h.length]!;
      v[i] = (byte / 255) * 2 - 1;
    }
    let n = 0;
    for (let i = 0; i < this.dim; i++) n += v[i]! * v[i]!;
    n = Math.sqrt(n);
    if (n !== 0) for (let i = 0; i < this.dim; i++) v[i] = v[i]! / n;
    return v;
  }
}

const EmbedResponse = z.object({ embeddings: z.array(z.array(z.number())) });

export class NomicHostedBackend extends EmbeddingBackend {
  readonly model: string;
  readonly dim: number;
  readonly batchSize: number;
  private readonly apiKey: string;

  static readonly DEFAULT_MODEL = "nomic-embed-text-v1.5";
  static readonly DEFAULT_DIM = 768;
  static readonly ENDPOINT = "https://api-atlas.nomic.ai/v1/embedding/text";

  constructor(opts?: { apiKey?: string; model?: string; dim?: number; batchSize?: number }) {
    super();
    this.apiKey = requireNomicKey(opts?.apiKey);
    this.model = opts?.model ?? NomicHostedBackend.DEFAULT_MODEL;
    this.dim = opts?.dim ?? NomicHostedBackend.DEFAULT_DIM;
    this.batchSize = opts?.batchSize ?? 32;
  }

  private async call(texts: string[]): Promise<Float32Array[]> {
    const resp = await fetch(NomicHostedBackend.ENDPOINT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({ model: this.model, texts }),
    });
    if (!resp.ok) {
      throw new Error(`Nomic API returned HTTP ${resp.status}: ${await resp.text()}`);
    }
    const payload = EmbedResponse.parse(await resp.json());
    return payload.embeddings.map((e) => Float32Array.from(e));
  }

  override embed(text: string): Promise<Float32Array>;
  override embed(texts: string[]): Promise<Float32Array[]>;
  override async embed(input: string | string[]): Promise<Float32Array | Float32Array[]> {
    if (!Array.isArray(input)) return (await this.call([input]))[0]!;
    const out: Float32Array[] = [];
    for (let i = 0; i < input.length; i += this.batchSize) {
      out.push(...(await this.call(input.slice(i, i + this.batchSize))));
    }
    return out;
  }
}

export class OllamaBackend extends EmbeddingBackend {
  readonly model: string;
  readonly dim: number;
  readonly baseUrl: string;
  readonly batchSize: number;

  constructor(opts: { model: string; dim: number; baseUrl?: string; batchSize?: number }) {
    super();
    this.model = opts.model;
    this.dim = opts.dim;
    this.baseUrl = opts.baseUrl ?? "http://localhost:11434";
    this.batchSize = opts.batchSize ?? 32;
  }

  private async call(input: string[]): Promise<Float32Array[]> {
    const url = this.baseUrl.replace(/\/+$/, "") + "/api/embed";
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: this.model, input }),
    });
    if (!resp.ok) {
      throw new Error(`Ollama returned HTTP ${resp.status}: ${await resp.text()}`);
    }
    const payload = EmbedResponse.parse(await resp.json());
    return payload.embeddings.map((e) => Float32Array.from(e));
  }

  override embed(text: string): Promise<Float32Array>;
  override embed(texts: string[]): Promise<Float32Array[]>;
  override async embed(input: string | string[]): Promise<Float32Array | Float32Array[]> {
    if (!Array.isArray(input)) return (await this.call([input]))[0]!;
    const out: Float32Array[] = [];
    for (let i = 0; i < input.length; i += this.batchSize) {
      out.push(...(await this.call(input.slice(i, i + this.batchSize))));
    }
    return out;
  }
}
