import { test, expect } from "bun:test";
import {
  DEFAULT_CONTENT_TYPES, PBTA_CONTENT_TYPES, YZE_CONTENT_TYPES,
} from "../src/content-types.ts";

test("content type sets", () => {
  for (const t of ["mechanic", "lore", "procedure", "boxed_text"] as const) {
    expect(DEFAULT_CONTENT_TYPES.has(t)).toBe(true);
  }
  expect(PBTA_CONTENT_TYPES.has("move")).toBe(true);
  expect(PBTA_CONTENT_TYPES.has("playbook")).toBe(true);
  for (const t of DEFAULT_CONTENT_TYPES) expect(PBTA_CONTENT_TYPES.has(t)).toBe(true);

  expect(YZE_CONTENT_TYPES.has("faction")).toBe(true);
  expect(YZE_CONTENT_TYPES.has("gear")).toBe(true);
  for (const t of DEFAULT_CONTENT_TYPES) expect(YZE_CONTENT_TYPES.has(t)).toBe(true);
});
