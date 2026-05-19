import { test, expect } from "bun:test";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { mkdtempSync, writeFileSync } from "node:fs";

test("cli ingests a markdown doc with mock embed + heuristic classify", async () => {
  const dir = mkdtempSync(join(tmpdir(), "tomerag-cli-"));
  const db = join(dir, "t.duckdb");
  const mdPath = join(dir, "doc.md");
  writeFileSync(mdPath, "# Moves\n\n## Iron Vow\n\n**When you swear upon iron**, roll +heart. On a 10+, succeed.\n");
  const cfgPath = join(dir, "source.json");
  writeFileSync(cfgPath, JSON.stringify({
    id: "iron", name: "Ironsworn", system: "PbtA", dbPath: db,
    embeddingModel: "mock", embeddingDim: 8, license: "cc_by",
    chunking: { minTokens: 1, maxTokens: 200, overflow: "paragraph", overlapTokens: 0 },
    contentTypes: ["mechanic", "move", "lore"],
  }));

  const proc = Bun.spawnSync([
    "bun", "run", join(import.meta.dir, "..", "src", "cli", "ingest.ts"),
    "--source-config", cfgPath, "--doc-id", "iron-core",
    "--document-type", "core_rules", "--path", mdPath,
    "--classify", "heuristic", "--embed", "mock",
  ], { cwd: dir });

  const out = proc.stdout.toString();
  expect(proc.exitCode).toBe(0);
  expect(out).toMatch(/inserted/i);
});
