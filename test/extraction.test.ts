import { test, expect } from "bun:test";
import { createHash } from "node:crypto";
import { mkdtempSync, writeFileSync, readFileSync, existsSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  MockExtractionBackend, CachingBackend, ExtractionBackend, splitPdftext,
} from "../src/backends/extraction.ts";
import type { PageText } from "../src/types.ts";

test("MockExtractionBackend", async () => {
  const pages: PageText[] = [
    { pageNum: 1, text: "# Iron Vow\nRoll +heart." },
    { pageNum: 2, text: "# Face Danger\nRoll +edge." },
  ];
  const b = new MockExtractionBackend(pages);
  const result = await b.extractPages("any/path.pdf");
  expect(result.length).toBe(2);
  expect(result[1]!.pageNum).toBe(2);

  const p = await b.extractPage("any/path.pdf", 2);
  expect(p.text).toBe("# Face Danger\nRoll +edge.");

  await expect(b.extractPage("any/path.pdf", 99)).rejects.toThrow();
});

test("splitPdftext splits on form-feed and skips blanks", () => {
  const p1 = splitPdftext("Page one content.\fPage two content.\f");
  expect(p1.length).toBe(2);
  expect(p1[0]).toEqual({ pageNum: 1, text: "Page one content." });
  expect(p1[1]).toEqual({ pageNum: 2, text: "Page two content." });

  const p2 = splitPdftext("Content\f\f\fMore content");
  expect(p2.length).toBe(2);
  expect(p2[0]).toEqual({ pageNum: 1, text: "Content" });
  expect(p2[1]).toEqual({ pageNum: 4, text: "More content" });
});

test("CachingBackend caches to disk and reuses", async () => {
  const dir = mkdtempSync(join(tmpdir(), "tomerag-cache-"));
  const pdf = join(dir, "test.pdf");
  writeFileSync(pdf, "fake pdf bytes for hashing");

  let calls = 0;
  class Counting extends ExtractionBackend {
    override async extractPages(): Promise<PageText[]> {
      calls++;
      return [{ pageNum: 1, text: "iron vow text" }, { pageNum: 2, text: "face danger text" }];
    }
  }
  const cache = new CachingBackend({ inner: new Counting(), cacheDir: join(dir, "cache") });

  const r1 = await cache.extractPages(pdf);
  expect(r1.length).toBe(2);
  expect(calls).toBe(1);

  const hash = createHash("sha256").update(readFileSync(pdf)).digest("hex");
  const cdir = join(dir, "cache", hash);
  expect(existsSync(join(cdir, "page_001.txt"))).toBe(true);
  expect(readFileSync(join(cdir, "page_001.txt"), "utf8")).toBe("iron vow text");

  await cache.extractPages(pdf); // second call: served from disk
  expect(calls).toBe(1);
});

test("CachingBackend keys by content, not filename", async () => {
  const dir = mkdtempSync(join(tmpdir(), "tomerag-cache-"));
  const a = join(dir, "a.pdf"); writeFileSync(a, "pdf content A");
  const b = join(dir, "b.pdf"); writeFileSync(b, "pdf content B");
  const cache = new CachingBackend({
    inner: new MockExtractionBackend([{ pageNum: 1, text: "content" }]),
    cacheDir: join(dir, "cache"),
  });
  await cache.extractPages(a);
  await cache.extractPages(b);
  const ha = createHash("sha256").update(readFileSync(a)).digest("hex");
  const hb = createHash("sha256").update(readFileSync(b)).digest("hex");
  expect(ha).not.toBe(hb);
  expect(readdirSync(join(dir, "cache")).sort()).toEqual([ha, hb].sort());
});
