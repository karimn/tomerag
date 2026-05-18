import { randomUUID } from "node:crypto";
import { resolve, extname } from "node:path";
import type {
  Chunk, DocumentType, RawChunk, Source,
} from "./types.ts";
import { chunkDocument } from "./chunker.ts";
import { contentHash, tokenCount } from "./tokenize.ts";
import type { EmbeddingBackend } from "./backends/embedding.ts";
import type { ClassifyBackend } from "./backends/classify.ts";
import type { ExtractionBackend } from "./backends/extraction.ts";
import { insertChunks } from "./storage.ts";

export interface IngestOptions {
  source: Source;
  path: string;
  docId: string;
  documentType: DocumentType;
  format?: "auto" | "markdown" | "pdf";
  embedBackend: EmbeddingBackend;
  classifyBackend: ClassifyBackend;
  extractionBackend?: ExtractionBackend;
}

export function injectPageMarker(pageText: string, pageNum: number): string {
  const marker = `<!-- page ${pageNum} -->`;
  const nl = pageText.indexOf("\n");
  const line1 = nl < 0 ? pageText : pageText.slice(0, nl);
  const rest = nl < 0 ? "" : pageText.slice(nl + 1);
  if (line1.trim().startsWith("#")) {
    return `${line1}\n${marker}\n${rest}`;
  }
  return `${marker}\n${pageText}`;
}

export function assignPages(chunks: RawChunk[]): RawChunk[] {
  return chunks.map((rc) => {
    const matches = [...rc.text.matchAll(/<!-- page (\d+) -->/g)];
    const page = matches.length === 0 ? "" : matches[matches.length - 1]![1]!;
    const clean = rc.text.replace(/<!-- page \d+ -->\n?/g, "").trim();
    return { headingPath: rc.headingPath, text: clean, chunkOrder: rc.chunkOrder, page };
  });
}

function searchText(r: RawChunk): string {
  if (r.headingPath.length === 0) return r.text;
  return `${r.headingPath.join(" > ")}\n\n${r.text}`;
}

export async function ingest(opts: IngestOptions): Promise<number> {
  let format = opts.format ?? "auto";
  if (format === "auto") {
    format = extname(opts.path).toLowerCase() === ".pdf" ? "pdf" : "markdown";
  }
  if (format === "pdf" && !opts.extractionBackend) {
    throw new Error("extractionBackend is required for format=pdf");
  }

  let docText: string;
  if (format === "pdf") {
    const pages = await opts.extractionBackend!.extractPages(opts.path);
    docText = pages.map((p) => injectPageMarker(p.text, p.pageNum)).join("\n\n");
  } else {
    docText = await Bun.file(opts.path).text();
  }

  let raws = chunkDocument(docText, opts.source.chunking);
  if (raws.length === 0) return 0;
  if (format === "pdf") raws = assignPages(raws);

  const classified = await opts.classifyBackend.classifyBatch(raws);
  const searchTexts = raws.map(searchText);
  const embeddings = await opts.embedBackend.embed(searchTexts);
  const absPath = resolve(opts.path);

  const chunks: Chunk[] = raws.map((r, i) => {
    const cls = classified[i]!;
    return {
      id: randomUUID(),
      sourceId: opts.source.id,
      docId: opts.docId,
      docPath: absPath,
      text: searchTexts[i]!,
      embedding: embeddings[i]!,
      embeddingModel: opts.source.embeddingModel,
      tokenCount: tokenCount(r.text),
      contentHash: contentHash(r.text),
      documentType: opts.documentType,
      system: opts.source.system,
      edition: "",
      page: r.page,
      headingPath: r.headingPath,
      chunkOrder: r.chunkOrder,
      parentId: null,
      contentType: cls.contentType,
      tags: cls.tags,
      moveTrigger: cls.moveTrigger,
      sceneType: cls.sceneType,
      encounterKey: cls.encounterKey,
      npcName: cls.npcName,
      license: opts.source.license,
    };
  });

  return insertChunks(opts.source, chunks);
}
