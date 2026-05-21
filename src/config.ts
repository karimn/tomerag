import { readFileSync } from "node:fs";
import { z } from "zod";

const ConfigSchema = z.object({
  anthropicApiKey: z.string().min(1).optional(),
  nomicApiKey: z.string().min(1).optional(),
});

export type TomeragConfig = z.infer<typeof ConfigSchema>;

const CONFIG_PATH = "./tomerag.config.json";
let cache: TomeragConfig | null = null;

export function loadConfig(): TomeragConfig {
  if (cache !== null) return cache;
  let raw: string;
  try {
    raw = readFileSync(CONFIG_PATH, "utf8");
  } catch {
    throw new Error(
      `TomeRAG config not found at ${CONFIG_PATH}. ` +
      `Create it with a JSON object like { "anthropicApiKey": "sk-ant-..." } ` +
      `(see tomerag.config.example.json).`,
    );
  }
  cache = ConfigSchema.parse(JSON.parse(raw));
  return cache;
}

/** Test-only: clear the module-local cache so a different cwd/file is re-read. */
export function resetConfigCache(): void {
  cache = null;
}

/** Resolve an Anthropic API key from explicit arg or config; throw if absent. */
export function requireAnthropicKey(explicit?: string): string {
  const key = explicit ?? loadConfig().anthropicApiKey;
  if (!key) {
    throw new Error(
      `Anthropic API key not set. Pass apiKey explicitly or add ` +
      `"anthropicApiKey" to ${CONFIG_PATH}.`,
    );
  }
  return key;
}

/**
 * Resolve a Nomic API key from explicit arg, NOMIC_API_KEY env var, or config;
 * throw if absent. The env var path lets you run without a tomerag.config.json.
 */
export function requireNomicKey(explicit?: string): string {
  if (explicit) return explicit;
  const fromEnv = process.env["NOMIC_API_KEY"];
  if (fromEnv) return fromEnv;
  let fromConfig: string | undefined;
  try {
    fromConfig = loadConfig().nomicApiKey;
  } catch {
    // config file absent — fall through to the unified error
  }
  if (fromConfig) return fromConfig;
  throw new Error(
    `Nomic API key not set. Pass apiKey explicitly, set NOMIC_API_KEY, ` +
    `or add "nomicApiKey" to ${CONFIG_PATH}.`,
  );
}
