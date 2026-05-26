import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  existsSync, mkdirSync, readFileSync, writeFileSync,
} from "node:fs";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Anthropic from "@anthropic-ai/sdk";
import type { PageText } from "../types.ts";
import { requireAnthropicKey } from "../config.ts";

export abstract class ExtractionBackend {
  abstract extractPages(pdfPath: string): Promise<PageText[]>;

  async extractPage(pdfPath: string, pageNum: number): Promise<PageText> {
    const pages = await this.extractPages(pdfPath);
    const found = pages.find((p) => p.pageNum === pageNum);
    if (!found) throw new Error(`page ${pageNum} not found in ${pdfPath}`);
    return found;
  }
}

export class MockExtractionBackend extends ExtractionBackend {
  constructor(private readonly pages: PageText[]) {
    super();
  }
  override async extractPages(_pdfPath: string): Promise<PageText[]> {
    return this.pages;
  }
}

/** Split `pdftotext` output on form-feed; skip blank pages but keep 1-based original index. */
export function splitPdftext(output: string): PageText[] {
  const raw = output.split("\f");
  const result: PageText[] = [];
  raw.forEach((r, i) => {
    const text = r.trim();
    if (text !== "") result.push({ pageNum: i + 1, text });
  });
  return result;
}

function run(bin: string, args: string[]): string {
  const r = spawnSync(bin, args, { encoding: "utf8", maxBuffer: 256 * 1024 * 1024 });
  if (r.error) throw new Error(`${bin} not found or failed: ${r.error.message}`);
  if (r.status !== 0) throw new Error(`${bin} exited ${r.status}: ${r.stderr}`);
  return r.stdout;
}

export class PopplerBackend extends ExtractionBackend {
  readonly firstPage: number | undefined;
  readonly lastPage: number | undefined;

  constructor(opts?: { firstPage?: number; lastPage?: number }) {
    super();
    this.firstPage = opts?.firstPage;
    this.lastPage = opts?.lastPage;
  }

  override async extractPages(pdfPath: string): Promise<PageText[]> {
    const args = ["-layout"];
    if (this.firstPage !== undefined) args.push("-f", String(this.firstPage));
    if (this.lastPage !== undefined) args.push("-l", String(this.lastPage));
    args.push(pdfPath, "-");
    return splitPdftext(run("pdftotext", args));
  }

  override async extractPage(pdfPath: string, pageNum: number): Promise<PageText> {
    const out = run("pdftotext", ["-layout", "-f", String(pageNum), "-l", String(pageNum), pdfPath, "-"]).trim();
    if (out === "") throw new Error(`page ${pageNum} is blank or out of range in ${pdfPath}`);
    return { pageNum, text: out };
  }
}

export class CachingBackend extends ExtractionBackend {
  private readonly inner: ExtractionBackend;
  private readonly cacheDir: string;

  constructor(opts: { inner: ExtractionBackend; cacheDir: string }) {
    super();
    this.inner = opts.inner;
    this.cacheDir = opts.cacheDir;
  }

  override async extractPages(pdfPath: string): Promise<PageText[]> {
    const hash = createHash("sha256").update(readFileSync(pdfPath)).digest("hex");
    const dir = join(this.cacheDir, hash);

    if (existsSync(join(dir, "page_001.txt"))) {
      const cached: PageText[] = [];
      let i = 1;
      for (;;) {
        const f = join(dir, `page_${String(i).padStart(3, "0")}.txt`);
        if (!existsSync(f)) break;
        cached.push({ pageNum: i, text: readFileSync(f, "utf8") });
        i++;
      }
      if (cached.length > 0) return cached;
    }

    const results = await this.inner.extractPages(pdfPath);
    mkdirSync(dir, { recursive: true });
    for (const pt of results) {
      writeFileSync(join(dir, `page_${String(pt.pageNum).padStart(3, "0")}.txt`), pt.text);
    }
    return results;
  }
}

const VISION_PROMPT =
  "This is page {PAGE} of an RPG rulebook. Extract all text exactly as written. " +
  "Format as markdown: use # headings for chapter/section titles, ## for subsections, " +
  "**bold** for move names and keywords, > blockquotes for sidebars and boxed text, " +
  "and markdown tables for any tabular content. Preserve the reading order. " +
  "Output only the extracted text, nothing else.";

function pLimit(concurrency: number) {
  let active = 0;
  const queue: (() => void)[] = [];
  const next = () => {
    active--;
    const job = queue.shift();
    if (job) job();
  };
  return <T>(fn: () => Promise<T>): Promise<T> =>
    new Promise<T>((resolve, reject) => {
      const runJob = () => {
        active++;
        fn().then(resolve, reject).finally(next);
      };
      if (active < concurrency) runJob();
      else queue.push(runJob);
    });
}

export class VisionBackend extends ExtractionBackend {
  readonly model: string;
  readonly concurrency: number;
  readonly dpi: number;
  readonly firstPage: number | undefined;
  readonly lastPage: number | undefined;
  private readonly client: Anthropic;

  constructor(opts?: { apiKey?: string; model?: string; concurrency?: number; dpi?: number; firstPage?: number; lastPage?: number }) {
    super();
    this.model = opts?.model ?? "claude-haiku-4-5-20251001";
    this.concurrency = opts?.concurrency ?? 5;
    this.dpi = opts?.dpi ?? 150;
    this.firstPage = opts?.firstPage;
    this.lastPage = opts?.lastPage;
    this.client = new Anthropic({ apiKey: requireAnthropicKey(opts?.apiKey) });
  }

  private pageCount(pdfPath: string): number {
    const out = run("pdfinfo", [pdfPath]);
    const m = out.match(/Pages:\s+(\d+)/);
    if (!m) throw new Error(`could not determine page count for ${pdfPath}`);
    return parseInt(m[1]!, 10);
  }

  private renderPage(pdfPath: string, pageNum: number): Buffer {
    const tmp = mkdtempSync(join(tmpdir(), "tomerag-ppm-"));
    try {
      const prefix = join(tmp, "page");
      run("pdftoppm", [
        "-r", String(this.dpi), "-f", String(pageNum), "-l", String(pageNum),
        "-png", "-singlefile", pdfPath, prefix,
      ]);
      const png = prefix + ".png";
      if (!existsSync(png)) throw new Error(`pdftoppm did not produce ${png} for page ${pageNum}`);
      return readFileSync(png);
    } finally {
      rmSync(tmp, { recursive: true, force: true });
    }
  }

  override async extractPage(pdfPath: string, pageNum: number): Promise<PageText> {
    const png = this.renderPage(pdfPath, pageNum);
    const resp = await this.client.messages.create({
      model: this.model,
      max_tokens: 4096,
      messages: [{
        role: "user",
        content: [
          { type: "image", source: { type: "base64", media_type: "image/png", data: png.toString("base64") } },
          { type: "text", text: VISION_PROMPT.replace("{PAGE}", String(pageNum)) },
        ],
      }],
    });
    const first = resp.content[0];
    const text = first && first.type === "text" ? first.text : "";
    return { pageNum, text };
  }

  override async extractPages(pdfPath: string): Promise<PageText[]> {
    const total = this.pageCount(pdfPath);
    const start = this.firstPage ?? 1;
    const end = this.lastPage ?? total;
    const limit = pLimit(this.concurrency);
    const pages = await Promise.all(
      Array.from({ length: end - start + 1 }, (_, i) => limit(() => this.extractPage(pdfPath, start + i))),
    );
    return pages.sort((a, b) => a.pageNum - b.pageNum);
  }
}
