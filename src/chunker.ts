import type { ChunkingConfig, Overflow, RawChunk, Section } from "./types.ts";
import { tokenCount, wsTokens } from "./tokenize.ts";

export function parseMarkdownSections(md: string): Section[] {
  const sections: Section[] = [];
  const stack: string[] = [];
  let buf: string[] = [];
  let started = false;

  const flush = () => {
    if (started) {
      const t = buf.join("\n").trim();
      if (t !== "") sections.push({ headingPath: [...stack], text: t });
    }
    buf = [];
  };

  for (const line of md.split("\n")) {
    const m = line.match(/^(#{1,6})\s+(.+?)\s*$/);
    if (m) {
      flush();
      const level = m[1]!.length;
      const title = m[2]!;
      while (stack.length >= level) stack.pop();
      while (stack.length < level - 1) stack.push("");
      stack.push(title);
      started = true;
    } else {
      buf.push(line);
    }
  }
  flush();
  return sections;
}

function greedyPack(units: string[], maxTokens: number): string[] {
  const out: string[] = [];
  let cur: string[] = [];
  let curTokens = 0;
  for (const raw of units) {
    const u = raw.trim();
    if (u === "") continue;
    const uTokens = tokenCount(u);
    if (curTokens + uTokens > maxTokens && curTokens > 0) {
      out.push(cur.join(" ").trim());
      cur = [];
      curTokens = 0;
    }
    if (uTokens > maxTokens) {
      if (curTokens > 0) {
        out.push(cur.join(" ").trim());
        cur = [];
        curTokens = 0;
      }
      out.push(u);
    } else {
      cur.push(u);
      curTokens += uTokens;
    }
  }
  if (curTokens > 0) out.push(cur.join(" ").trim());
  return out;
}

function applyOverlap(pieces: string[], overlapTokens: number): string[] {
  if (overlapTokens <= 0 || pieces.length <= 1) return pieces;
  const out: string[] = [pieces[0]!];
  for (let i = 1; i < pieces.length; i++) {
    const prev = wsTokens(pieces[i - 1]!);
    const tail = prev.slice(Math.max(prev.length - overlapTokens, 0));
    out.push(tail.join(" ") + " " + pieces[i]!);
  }
  return out;
}

export function splitToTokenBudget(
  text: string,
  opts: { maxTokens: number; overlapTokens?: number; overflow?: Overflow },
): string[] {
  const maxTokens = opts.maxTokens;
  const overlapTokens = opts.overlapTokens ?? 0;
  let overflow: Overflow = opts.overflow ?? "paragraph";

  if (tokenCount(text) <= maxTokens) return [text];

  if (overflow === "paragraph") {
    const pieces = greedyPack(text.split(/\n\s*\n/), maxTokens);
    if (pieces.every((p) => tokenCount(p) <= maxTokens)) return applyOverlap(pieces, overlapTokens);
    overflow = "sentence";
  }
  if (overflow === "sentence") {
    const pieces = greedyPack(text.split(/(?<=[.!?])\s+/), maxTokens);
    if (pieces.every((p) => tokenCount(p) <= maxTokens)) return applyOverlap(pieces, overlapTokens);
    overflow = "token";
  }

  const docLines = text.split("\n");
  let curLines: string[] = [];
  let curTokens = 0;
  const pieces: string[] = [];
  let i = 0;
  while (i < docLines.length) {
    const line = docLines[i]!;
    if (line.trimStart().startsWith("|")) {
      const tblLines: string[] = [];
      let tblTokens = 0;
      while (i < docLines.length && docLines[i]!.trimStart().startsWith("|")) {
        tblLines.push(docLines[i]!);
        tblTokens += tokenCount(docLines[i]!);
        i++;
      }
      if (curTokens + tblTokens > maxTokens && curTokens > 0) {
        pieces.push(curLines.join("\n").trim());
        curLines = [];
        curTokens = 0;
      }
      curLines.push(...tblLines);
      curTokens += tblTokens;
      continue;
    }
    const lineTokens = tokenCount(line);
    if (curTokens + lineTokens > maxTokens && curTokens > 0) {
      pieces.push(curLines.join("\n").trim());
      curLines = [];
      curTokens = 0;
    }
    if (lineTokens > maxTokens) {
      const lineToks = wsTokens(line);
      let j = 0;
      while (j < lineToks.length) {
        const k = Math.min(j + maxTokens, lineToks.length);
        pieces.push(lineToks.slice(j, k).join(" "));
        j = k;
      }
      curLines = [];
      curTokens = 0;
    } else {
      curLines.push(line);
      curTokens += lineTokens;
    }
    i++;
  }
  if (curTokens > 0) pieces.push(curLines.join("\n").trim());
  return applyOverlap(pieces, overlapTokens);
}

export function chunkDocument(md: string, cfg: ChunkingConfig): RawChunk[] {
  const sections = parseMarkdownSections(md);
  const out: RawChunk[] = [];
  let pending: Section | null = null;

  const emit = (sec: Section) => {
    const pieces = splitToTokenBudget(sec.text, {
      maxTokens: cfg.maxTokens,
      overlapTokens: cfg.overlapTokens,
      overflow: cfg.overflow,
    });
    for (const p of pieces) {
      out.push({ headingPath: sec.headingPath, text: p, chunkOrder: out.length, page: "" });
    }
  };

  for (const sec of sections) {
    if (tokenCount(sec.text) < cfg.minTokens) {
      pending = pending === null
        ? sec
        : { headingPath: pending.headingPath, text: pending.text + "\n\n" + sec.text };
      continue;
    }
    if (pending !== null) {
      emit({ headingPath: pending.headingPath, text: pending.text + "\n\n" + sec.text });
      pending = null;
    } else {
      emit(sec);
    }
  }
  if (pending !== null) emit(pending);
  return out;
}
