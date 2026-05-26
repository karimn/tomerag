import { initializeStore, sourceStats, similaritySearch, bm25Search, withConnection } from "@tomerag/client";
import type { Chunk } from "@tomerag/client";
import type { Source } from "./types.ts";

export { initializeStore, sourceStats, similaritySearch, bm25Search } from "@tomerag/client";

function embeddingLiteral(emb: Float32Array, dim: number): string {
  return `[${Array.from(emb).join(",")}]::FLOAT[${dim}]`;
}

export async function insertChunks(src: Source, chunks: Chunk[]): Promise<number> {
  if (chunks.length === 0) return 0;
  let inserted = 0;
  await withConnection(src.dbPath, async (conn) => {
    await conn.run("LOAD vss;");
    await conn.run("LOAD fts;");
    await conn.run("BEGIN;");
    try {
      for (const c of chunks) {
        const existing = await conn.runAndReadAll(
          "SELECT 1 FROM chunks WHERE doc_id = ? AND content_hash = ? LIMIT 1",
          [c.docId, c.contentHash],
        );
        if (existing.getRowObjectsJS().length > 0) continue;
        await conn.run(
          `INSERT INTO chunks VALUES
           (?, ?, ?, ?, ?, ${embeddingLiteral(c.embedding, src.embeddingDim)},
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
          [
            c.id, c.sourceId, c.docId, c.docPath, c.text,
            c.embeddingModel, c.tokenCount, c.contentHash, c.documentType,
            c.system, c.edition, c.page, JSON.stringify(c.headingPath),
            c.chunkOrder, c.parentId, c.contentType, JSON.stringify(c.tags),
            c.moveTrigger, c.sceneType, c.encounterKey, c.npcName, c.license,
          ],
        );
        inserted++;
      }
      await conn.run("COMMIT;");
      if (inserted > 0) {
        await conn.run("PRAGMA create_fts_index('chunks', 'id', 'text', stemmer='porter', overwrite=1);");
      }
    } catch (e) {
      try { await conn.run("ROLLBACK;"); } catch { /* ignore */ }
      throw e;
    }
  });
  return inserted;
}
