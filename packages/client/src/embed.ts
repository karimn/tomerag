export async function ollamaEmbed(
  texts: string[],
  opts: { model: string; baseUrl?: string; batchSize?: number },
): Promise<Float32Array[]> {
  const baseUrl = (opts.baseUrl ?? "http://localhost:11434").replace(/\/+$/, "");
  const batchSize = opts.batchSize ?? 32;
  const results: Float32Array[] = [];

  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize);
    const resp = await fetch(`${baseUrl}/api/embed`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: opts.model, input: batch }),
    });
    if (!resp.ok) throw new Error(`Ollama returned HTTP ${resp.status}: ${await resp.text()}`);
    const payload = (await resp.json()) as { embeddings: number[][] };
    results.push(...payload.embeddings.map((e) => Float32Array.from(e)));
  }
  return results;
}
