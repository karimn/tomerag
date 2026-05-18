import type { Classification, ContentType, RawChunk } from "../types.ts";

export abstract class ClassifyBackend {
  abstract classify(input: { text: string; headingPath: string[] }): Promise<Classification>;

  async classifyBatch(raws: RawChunk[]): Promise<Classification[]> {
    const out: Classification[] = [];
    for (const r of raws) out.push(await this.classify({ text: r.text, headingPath: r.headingPath }));
    return out;
  }
}

export class MockClassifyBackend extends ClassifyBackend {
  private readonly contentType: ContentType;
  private readonly tags: string[];

  constructor(opts?: { contentType?: ContentType; tags?: string[] }) {
    super();
    this.contentType = opts?.contentType ?? "mechanic";
    this.tags = opts?.tags ?? [];
  }

  override async classify(): Promise<Classification> {
    return {
      contentType: this.contentType,
      tags: [...this.tags],
      moveTrigger: null,
      sceneType: null,
      encounterKey: null,
      npcName: null,
    };
  }
}

const MOVE_TRIGGER_PAT = /\*\*when\s+([^*]+?)\*\*/i;
const TABLE_PAT = /^\s*\|.*\|\s*$/gm;
const STAT_LINE_PAT = /\b(hp|hit points|armor|ac|attack)\b[^\n]*\d/i;
const TABLE_HEADING_PAT = /tables?|oracle|random/i;
const BESTIARY_HEADING_PAT = /bestiary|stat ?block|npcs?|monsters?/i;
const LORE_HEADING_PAT = /world|geography|history|cult|faction|lore/i;
const GM_HEADING_PAT = /running|gm ?(guide|advice)|how to run/i;

export class HeuristicBackend extends ClassifyBackend {
  override async classify(input: { text: string; headingPath: string[] }): Promise<Classification> {
    const { text, headingPath } = input;
    const joinedHeading = headingPath.join(" / ").toLowerCase();
    const base = { tags: [] as string[], moveTrigger: null, sceneType: null, encounterKey: null, npcName: null };

    const m = text.match(MOVE_TRIGGER_PAT);
    if (m) {
      return { ...base, contentType: "move", moveTrigger: m[1]!.trim() };
    }

    const tableMatches = text.match(TABLE_PAT);
    if ((tableMatches?.length ?? 0) >= 2 || TABLE_HEADING_PAT.test(joinedHeading)) {
      return { ...base, contentType: "table" };
    }

    if (BESTIARY_HEADING_PAT.test(joinedHeading) && STAT_LINE_PAT.test(text)) {
      const last = headingPath.length === 0 ? null : headingPath[headingPath.length - 1]!.trim();
      return { ...base, contentType: "stat_block", npcName: last && last !== "" ? last : null };
    }

    if (GM_HEADING_PAT.test(joinedHeading)) return { ...base, contentType: "gm_guidance" };
    if (LORE_HEADING_PAT.test(joinedHeading)) return { ...base, contentType: "lore" };
    return { ...base, contentType: "mechanic" };
  }
}
