import { test, expect, afterEach } from "bun:test";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { loadConfig, resetConfigCache } from "../src/config.ts";

const origCwd = process.cwd();
afterEach(() => {
  process.chdir(origCwd);
  resetConfigCache();
});

test("loadConfig reads and validates ./tomerag.config.json", () => {
  const dir = mkdtempSync(join(tmpdir(), "tomerag-cfg-"));
  writeFileSync(join(dir, "tomerag.config.json"), JSON.stringify({ anthropicApiKey: "sk-ant-test" }));
  process.chdir(dir);
  resetConfigCache();
  expect(loadConfig().anthropicApiKey).toBe("sk-ant-test");
  rmSync(dir, { recursive: true, force: true });
});

test("loadConfig throws a helpful error when the file is missing", () => {
  const dir = mkdtempSync(join(tmpdir(), "tomerag-cfg-"));
  process.chdir(dir);
  resetConfigCache();
  expect(() => loadConfig()).toThrow(/tomerag\.config\.json/);
  rmSync(dir, { recursive: true, force: true });
});
