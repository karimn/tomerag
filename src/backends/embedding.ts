import { createHash } from "node:crypto";
import { z } from "zod";

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

const OllamaResponse = z.object({ embeddings: z.array(z.array(z.number())) });

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
    const payload = OllamaResponse.parse(await resp.json());
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
