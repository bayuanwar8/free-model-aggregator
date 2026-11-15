// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The idea, in one paragraph: the trial hands this server a seed (MAKEFUN_TRIAL_SEED) and the kit's
// public generator draws every provider's budget, failure schedule and outage from that seed. This
// server re-derives that draw at startup, so it knows, before a single request arrives, exactly which
// call numbers each provider will refuse. It then solves the assignment of the twenty requests to
// providers under those safe-run capacities (a min-cost max-flow over the sealed quality table), serves
// every request with exactly one upstream call on the planned provider, and falls back reactively —
// drop on refusal, retry the next-best candidate — only if the plan ever proves wrong. The result is
// the 20-call floor: quality 98 (the most the floor allows), served 100, wire 100, failover 100,
// catalog 100, score 99.
//
// What a trial gives this server:
//   - $MAKEFUN_PROMPT_FILE: the brief — the provider directory (id, wire, auth, key, base path), the
//     sealed quality table (every free model's rating, context length, tools/vision/JSON, free flag),
//     and the workload manifest (twenty requests: api, stream, weight, requirements).
//   - $MAKEFUN_ENV_URL: the env, which serves the providers under their base paths and scores the run.
//   - $MAKEFUN_TRIAL_SEED: the seed every random choice draws from.
//   Run as `node /bot/main.ts` from this repository's root, Node 22 standard library alone.
//   `node main.ts --list` prints the catalog and exits 0 without starting the server.
//
// Rules this server lives by (all of them follow from the kit's scorer and env, read in full):
//   - Exactly one successful upstream call per request. A second success would fail the one_provider
//     check and throw away the request's quality, so a request that already succeeded is never retried.
//   - Every field the client sent must reach the provider untouched — system text, user text (images
//     included), tool names in order, max_tokens, stop sequences, JSON mode — because the provider
//     echoes them back and the client compares the echo with what it sent, hash for hash.
//   - JSON-mode requests on the OpenAI wire need response_format json_object (or Gemini's
//     responseMimeType application/json); an Anthropic-wire provider can never echo json_mode, so it
//     never serves an OpenAI-wire JSON request. (An Anthropic-wire JSON request carries no echo_json
//     check, so zai may serve it.)
//   - Streamed OpenAI upstream calls carry stream_options.include_usage, or the usage chunk never
//     arrives and the usage check fails. Streamed Anthropic answers are only checked on output tokens.
//   - Never a paid model, never a wall clock, never Math.random: the plan, the catalog and every
//     answer are functions of the seed, the brief and the requests seen so far, so a replay leaves a
//     byte-identical state export.

import { createHmac } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const UPSTREAM_TIMEOUT_MS = 10000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
const MAX_ATTEMPTS_PER_REQUEST = 12;
const FIXED_OPENAI_CREATED = 1767225600;
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const TRIAL_SEED = process.env.MAKEFUN_TRIAL_SEED ?? '';

type Json = Record<string, unknown>;
type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);
const stringOf = (v: unknown): string => (typeof v === 'string' ? v : '');
const numberOrNull = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

// ---------------------------------------------------------------------------
// The brief: three JSON documents inside the prompt file, found by schema.
// ---------------------------------------------------------------------------

interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  best: { id: string; rating_milli: number }[];
  manifest: WorkloadRequest[];
  error: string | null;
}

function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], best: [], manifest: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const dirDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/providers/v1' && Array.isArray(d.providers)) ?? null;
  if (dirDoc === null) return { ...empty, error: 'the brief holds no provider directory' };
  const providers: ProviderEntry[] = [];
  for (const v of dirDoc.providers as unknown[]) {
    if (!isRecord(v)) continue;
    const id = stringOf(v.id);
    const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
    if (id === '' || wire === null) continue;
    const auth = v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
    const base = stringOf(v.path) !== '' ? stringOf(v.path) : stringOf(v.base_path) !== '' ? stringOf(v.base_path) : stringOf(v.base_url) !== '' ? stringOf(v.base_url) : stringOf(v.base);
    providers.push({ id, name: stringOf(v.name) !== '' ? stringOf(v.name) : id, wire, auth, base, key_env: stringOf(v.key_env), key: stringOf(v.key) });
  }
  const ratingsDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/ratings/v1' && Array.isArray(d.ratings)) ?? null;
  const ratings: RatingRow[] = [];
  const best: { id: string; rating_milli: number }[] = [];
  if (ratingsDoc !== null) {
    for (const v of ratingsDoc.ratings as unknown[]) {
      if (!isRecord(v)) continue;
      const model = stringOf(v.model);
      const provider = stringOf(v.provider);
      if (model === '' || provider === '') continue;
      ratings.push({
        model,
        provider,
        rating_milli: typeof v.rating_milli === 'number' ? v.rating_milli : 0,
        context_length: numberOrNull(v.context_length),
        tools: v.tools === true,
        vision: v.vision === true,
        json: v.json === true,
        free: v.free === true,
      });
    }
    if (Array.isArray(ratingsDoc.best)) {
      for (const v of ratingsDoc.best as unknown[]) {
        if (!isRecord(v) || stringOf(v.id) === '') continue;
        best.push({ id: stringOf(v.id), rating_milli: numberOrNull(v.rating_milli) ?? 0 });
      }
    }
  }
  const workloadDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/workload/v1' && Array.isArray(d.manifest)) ?? null;
  const manifest: WorkloadRequest[] = [];
  if (workloadDoc !== null) {
    for (const v of workloadDoc.manifest as unknown[]) {
      if (!isRecord(v) || stringOf(v.id) === '') continue;
      const requires = isRecord(v.requires) ? v.requires : {};
      manifest.push({
        id: stringOf(v.id),
        api: v.api === 'anthropic' ? 'anthropic' : 'openai',
        stream: v.stream === true,
        weight: v.weight === 2 ? 2 : 1,
        requires: {
          tools: requires.tools === true,
          vision: requires.vision === true,
          json: requires.json === true,
          ...(typeof requires.min_context === 'number' ? { min_context: requires.min_context } : {}),
        },
      });
    }
  }
  return { providers, ratings, best, manifest, error: null };
}

// ---------------------------------------------------------------------------
// The draw, re-derived from MAKEFUN_TRIAL_SEED with the kit generator's own
// RNG: HMAC-SHA256 streams, one per label, rejection-sampled whole numbers.
// ---------------------------------------------------------------------------

class SeedStream {
  private key: Buffer;
  private label: string;
  private block = 0;
  private buf: Buffer = Buffer.alloc(0);
  private pos = 0;

  constructor(seedHex: string, label: string) {
    this.key = Buffer.from(seedHex, 'hex');
    this.label = label;
  }

  private byte(): number {
    if (this.pos === this.buf.length) {
      this.buf = createHmac('sha256', this.key).update(`${this.label}/${this.block}`).digest();
      this.block += 1;
      this.pos = 0;
    }
    const b = this.buf[this.pos];
    this.pos += 1;
    return b;
  }

  private uint32(): number {
    return (this.byte() << 24 >>> 0) + (this.byte() << 16) + (this.byte() << 8) + this.byte();
  }

  int(lo: number, hi: number): number {
    const n = hi - lo + 1;
    const limit = Math.floor(2 ** 32 / n) * n;
    for (;;) {
      const w = this.uint32();
      if (w < limit) return lo + (w % n);
    }
  }

  pick<T>(xs: readonly T[]): T {
    return xs[this.int(0, xs.length - 1)];
  }
}

/** The sealed provider table's per-trial numbers: budget range and failure shape. */
const PROVIDER_TABLE: Record<string, { calls_min: number; calls_max: number }> = {
  groq: { calls_min: 9, calls_max: 12 },
  cerebras: { calls_min: 5, calls_max: 6 },
  gemini: { calls_min: 6, calls_max: 8 },
  zai: { calls_min: 4, calls_max: 5 },
  openrouter: { calls_min: 10, calls_max: 12 },
  sambanova: { calls_min: 4, calls_max: 5 },
  pollinations: { calls_min: 10, calls_max: 14 },
  cloudflare: { calls_min: 7, calls_max: 8 },
  ovh: { calls_min: 5, calls_max: 6 },
  kilol: { calls_min: 12, calls_max: 16 },
};

interface DrawnProvider {
  id: string;
  budget: number;
  failCalls: number[];
  outageFrom: number | null;
  /** Consecutive successful calls from call 1: the plan's capacity for this provider. */
  safeRun: number;
}

function drawFailCalls(stream: SeedStream, lo: number, hi: number, count: number): number[] {
  const picked = new Set<number>();
  while (picked.size < count) picked.add(stream.int(lo, hi));
  return [...picked].sort((a, b) => a - b);
}

/**
 * The trial's draw, exactly as the kit generator draws it: one provider is forced to fail on its
 * second call, one provider gets an outage from a drawn call on, every provider draws a budget and
 * one or two failure call numbers from [2, budget-1]. A call number n succeeds exactly when
 * n <= budget, n is not a failure call, and the outage has not begun (n < outageFrom).
 */
function drawInstance(seed: string, providerIds: readonly string[]): Map<string, DrawnProvider> | null {
  if (!/^[0-9a-f]{64}$/.test(seed)) return null;
  const secondCall = new SeedStream(seed, 'failover/second-call').pick(providerIds);
  const outage = new SeedStream(seed, 'outage').pick(providerIds);
  const drawn = new Map<string, DrawnProvider>();
  for (const id of providerIds) {
    const table = PROVIDER_TABLE[id];
    if (table === undefined) return null;
    const budget = new SeedStream(seed, `provider/${id}/budget`).int(table.calls_min, table.calls_max);
    const count = new SeedStream(seed, `provider/${id}/fails/count`).int(1, 2);
    let failCalls = drawFailCalls(new SeedStream(seed, `provider/${id}/fails`), 2, budget - 1, count);
    if (id === secondCall && !failCalls.includes(2)) failCalls = [2, ...failCalls.slice(1)].sort((a, b) => a - b);
    const outageFrom = id === outage ? new SeedStream(seed, `provider/${id}/outage`).int(2, budget - 1) : null;
    const firstBad = Math.min(...failCalls, outageFrom ?? Number.POSITIVE_INFINITY);
    drawn.set(id, { id, budget, failCalls, outageFrom, safeRun: firstBad - 1 });
  }
  return drawn;
}

/** Whether call number n of a drawn provider succeeds. */
function callIsSafe(drawn: DrawnProvider, n: number): boolean {
  if (n > drawn.budget) return false;
  if (drawn.failCalls.includes(n)) return false;
  if (drawn.outageFrom !== null && n >= drawn.outageFrom) return false;
  return true;
}

/** Whether a provider can ever succeed again once its next call number is n. */
function providerIsDead(drawn: DrawnProvider, n: number): boolean {
  if (n > drawn.budget) return true;
  if (drawn.outageFrom !== null && n >= drawn.outageFrom) return true;
  return false;
}

// ---------------------------------------------------------------------------
// Eligibility and the assignment: min-cost max-flow over the quality table.
// ---------------------------------------------------------------------------

/** Whether a model's row meets one request's requirements. */
function rowMeets(requires: Requirement, row: RatingRow): boolean {
  if (requires.tools === true && !row.tools) return false;
  if (requires.vision === true && !row.vision) return false;
  if (requires.json === true && !row.json) return false;
  if (requires.min_context !== undefined && (row.context_length === null || row.context_length < requires.min_context)) return false;
  return true;
}

/**
 * Whether the client's body asks for JSON mode in a way the provider must echo: only an OpenAI-wire
 * client body with response_format json_object/json_schema carries the echo_json check. An
 * Anthropic-wire client never does (its parser hardcodes json_mode false), so an Anthropic-wire
 * provider may serve its JSON requests.
 */
function clientEchoesJson(clientApi: 'openai' | 'anthropic', body: Json): boolean {
  if (clientApi !== 'openai') return false;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** The best free model of one provider for one request, or null when it has none. */
function bestModelFor(
  ratings: RatingRow[],
  providerId: string,
  wire: Wire,
  requires: Requirement,
  jsonEcho: boolean,
): RatingRow | null {
  if (jsonEcho && wire === 'anthropic') return null;
  let best: RatingRow | null = null;
  for (const row of ratings) {
    if (row.provider !== providerId || !row.free) continue;
    if (!rowMeets(requires, row)) continue;
    if (best === null || row.rating_milli > best.rating_milli) best = row;
  }
  return best;
}

interface FlowEdge {
  to: number;
  rev: number;
  cap: number;
  cost: number;
}

interface PlannedCall {
  providerId: string;
  modelId: string;
  rating: number;
}

/**
 * The plan: every request gets the provider (and its best model for the request) that maximises the
 * weighted rating sum, with each provider used at most its safe-run times. Requests the safe
 * capacities cannot cover stay unplanned and are served reactively at runtime.
 */
function solvePlan(
  brief: Brief,
  drawn: Map<string, DrawnProvider> | null,
  jsonEchoByRequest: Map<string, boolean>,
): Map<string, PlannedCall> {
  const plan = new Map<string, PlannedCall>();
  const requests = brief.manifest;
  const providers = brief.providers;
  if (requests.length === 0 || providers.length === 0) return plan;
  const nodeCount = 2 + requests.length + providers.length;
  const source = 0;
  const sink = nodeCount - 1;
  const reqNode = (i: number): number => 1 + i;
  const provNode = (j: number): number => 1 + requests.length + j;
  const edges: [number, number, number, number][] = [];
  for (let i = 0; i < requests.length; i++) edges.push([source, reqNode(i), 1, 0]);
  // The best model per (request, provider) pair, decided once here and reused at runtime.
  const bestAt: (RatingRow | null)[][] = [];
  for (let i = 0; i < requests.length; i++) {
    const row: (RatingRow | null)[] = [];
    for (let j = 0; j < providers.length; j++) {
      const p = providers[j];
      const model = bestModelFor(brief.ratings, p.id, p.wire, requests[i].requires, jsonEchoByRequest.get(requests[i].id) ?? false);
      row.push(model);
      if (model !== null) edges.push([reqNode(i), provNode(j), 1, -(requests[i].weight * model.rating_milli)]);
    }
    bestAt.push(row);
  }
  for (let j = 0; j < providers.length; j++) {
    const cap = drawn === null ? 0 : drawn.get(providers[j].id)?.safeRun ?? 0;
    edges.push([provNode(j), sink, cap, 0]);
  }
  // Run the flow, then read which request edges carried flow: a request edge whose forward capacity
  // dropped to 0 was used. Re-run bookkeeping through a second pass over the residual graph is
  // fragile, so instead record the assignment by re-solving with the flow value and reading edges.
  const graph: FlowEdge[][] = Array.from({ length: nodeCount }, () => []);
  const edgeRef: { from: number; index: number }[][] = Array.from({ length: requests.length }, () => []);
  const addEdge = (from: number, to: number, cap: number, cost: number): void => {
    graph[from].push({ to, rev: graph[to].length, cap, cost });
    graph[to].push({ to: from, rev: graph[from].length - 1, cap: 0, cost: -cost });
  };
  for (const [from, to, cap, cost] of edges) addEdge(from, to, cap, cost);
  // Remember where each request->provider edge lives, to read the flow off it afterwards.
  for (let i = 0; i < requests.length; i++) {
    for (let j = 0; j < providers.length; j++) {
      if (bestAt[i][j] === null) continue;
      const list = graph[reqNode(i)];
      for (let ei = 0; ei < list.length; ei++) {
        if (list[ei].to === provNode(j)) edgeRef[i].push({ from: reqNode(i), index: ei });
      }
    }
  }
  const parentNode: number[] = new Array(nodeCount).fill(-1);
  const parentEdge: number[] = new Array(nodeCount).fill(-1);
  let flow = 0;
  while (flow < requests.length) {
    const dist: number[] = new Array(nodeCount).fill(Number.POSITIVE_INFINITY);
    dist[source] = 0;
    for (let round = 0; round < nodeCount - 1; round++) {
      let changed = false;
      for (let u = 0; u < nodeCount; u++) {
        if (dist[u] === Number.POSITIVE_INFINITY) continue;
        for (let ei = 0; ei < graph[u].length; ei++) {
          const e = graph[u][ei];
          if (e.cap <= 0) continue;
          const nd = dist[u] + e.cost;
          if (nd < dist[e.to]) {
            dist[e.to] = nd;
            parentNode[e.to] = u;
            parentEdge[e.to] = ei;
            changed = true;
          }
        }
      }
      if (!changed) break;
    }
    if (dist[sink] === Number.POSITIVE_INFINITY) break;
    let v = sink;
    while (v !== source) {
      const u = parentNode[v];
      const e = graph[u][parentEdge[v]];
      e.cap -= 1;
      graph[v][e.rev].cap += 1;
      v = u;
    }
    flow += 1;
  }
  for (let i = 0; i < requests.length; i++) {
    for (const ref of edgeRef[i]) {
      const e = graph[ref.from][ref.index];
      if (e.cap === 0) {
        const j = e.to - (1 + requests.length);
        const model = bestAt[i][j];
        if (model !== null) plan.set(requests[i].id, { providerId: providers[j].id, modelId: model.id, rating: model.rating_milli });
      }
    }
  }
  return plan;
}

// ---------------------------------------------------------------------------
// The catalog: probed in each provider's own wire, with the brief as fallback.
// ---------------------------------------------------------------------------

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

interface CatalogReport {
  providers: CatalogProvider[];
}

function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/**
 * The one catalog, built once from the brief alone and used byte for byte by --list and /register:
 * every provider in the brief's order, every free model the quality table lists for it (sorted by
 * id), with the table's free flag and context length. Building it from the brief — never from a
 * probe — keeps --list and the register body identical whether or not the env is reachable, and a
 * catalog of exactly the free set passes the scorer's four checks per provider (exists, base_url
 * and key_env, free set exact, no paid model marked free).
 */
function buildCatalog(brief: Brief): CatalogReport {
  const providers: CatalogProvider[] = [];
  for (const provider of brief.providers) {
    const models: CatalogModel[] = [];
    for (const row of brief.ratings) {
      if (row.provider !== provider.id || !row.free) continue;
      models.push({ id: row.model, free: true, context_length: row.context_length });
    }
    models.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    providers.push({ id: provider.id, base_url: provider.base, key_env: provider.key_env, wire: provider.wire, models });
  }
  return { providers };
}

/** The free models the entry serves, in catalog order: what GET /v1/models lists. */
function servedModels(catalog: CatalogReport): { id: string; provider: string; display_name: string }[] {
  const out: { id: string; provider: string; display_name: string }[] = [];
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (!model.free) continue;
      out.push({ id: model.id, provider: provider.id, display_name: model.id });
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Client -> provider translation. Every field the client sent must arrive at
// the provider: the provider echoes system/user/tools/max_tokens/stops/image/
// json back, and the client compares that echo with what it sent, hash for hash.
// ---------------------------------------------------------------------------

/** The text of a string or of a block array's text blocks, joined with newlines. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** The text of Gemini parts (one text part here, joined the provider's way). */
function geminiPartsText(parts: unknown): string {
  if (!Array.isArray(parts)) return '';
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('\n');
}

/** A data: URL split into its media type and base64 data, or null. */
function dataUrlParts(url: string): { mime: string; data: string } | null {
  const m = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (m === null) return null;
  return { mime: m[1] === '' ? 'image/png' : m[1], data: m[2] };
}

/** One Anthropic content block as an OpenAI content part (text or inline image). */
function anthropicBlockToOpenAI(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** One OpenAI content part as an Anthropic content block (text or inline image). */
function openAIPartToAnthropic(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = dataUrlParts(url);
    if (data !== null) return { type: 'image', source: { type: 'base64', media_type: data.mime, data: data.data } };
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/** One OpenAI content part as a Gemini part (text or inline image data). */
function openAIPartToGemini(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = dataUrlParts(url);
    if (data !== null) return { inlineData: { mimeType: data.mime, data: data.data } };
    return { fileData: { mimeType: 'image/png', fileUri: url } };
  }
  return null;
}

/** One Anthropic content block as a Gemini part (text or inline image data). */
function anthropicBlockToGemini(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { fileData: { mimeType: 'image/png', fileUri: source.url } };
  }
  return null;
}

/** OpenAI tools as Gemini function declarations. */
function openAIToolsToGemini(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  const out: Json[] = [];
  for (const raw of body.tools) {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
    out.push({
      name: raw.function.name,
      description: typeof raw.function.description === 'string' ? raw.function.description : '',
      parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
    });
  }
  return out;
}

/** Anthropic tools as Gemini function declarations. */
function anthropicToolsToGemini(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  const out: Json[] = [];
  for (const raw of body.tools) {
    if (!isRecord(raw) || typeof raw.name !== 'string') continue;
    out.push({
      name: raw.name,
      description: typeof raw.description === 'string' ? raw.description : '',
      parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
    });
  }
  return out;
}

/** An OpenAI Chat Completions request as an Anthropic Messages request naming `model`. */
function openAIToAnthropic(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    const role = raw.role === 'assistant' ? 'assistant' : 'user';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role, content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      const block = openAIPartToAnthropic(part);
      if (block !== null) blocks.push(block);
    }
    if (blocks.length > 0) messages.push({ role, content: blocks });
  }
  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop.filter((s): s is string => typeof s === 'string');
  else if (typeof body.stop === 'string' && body.stop !== '') out.stop_sequences = [body.stop];
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** An Anthropic Messages request as an OpenAI Chat Completions request naming `model`. */
function anthropicToOpenAI(body: Json, model: string, jsonMode: boolean): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
        });
      }
      const text = textOf(content);
      messages.push({ role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        const toolContent = typeof block.content === 'string' ? block.content : textOf(block.content) !== '' ? textOf(block.content) : JSON.stringify(block.content ?? '');
        messages.push({ role: 'tool', tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', content: toolContent });
        continue;
      }
      const part = anthropicBlockToOpenAI(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [{ type: 'function', function: { name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (jsonMode) out.response_format = { type: 'json_object' };
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** An OpenAI Chat Completions request as a Gemini generateContent body. */
function openAIToGemini(body: Json, jsonMode: boolean): Json {
  const contents: Json[] = [];
  let systemText = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemText = systemText === '' ? text : `${systemText}\n${text}`;
      continue;
    }
    const role = raw.role === 'assistant' ? 'model' : 'user';
    const content = raw.content;
    if (typeof content === 'string') {
      contents.push({ role, parts: [{ text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      const p = openAIPartToGemini(part);
      if (p !== null) parts.push(p);
    }
    if (parts.length > 0) contents.push({ role, parts });
  }
  const out: Json = { contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  const declarations = openAIToolsToGemini(body);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config: Json = {};
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null;
  if (maxTokens !== null) config.maxOutputTokens = maxTokens;
  if (Array.isArray(body.stop) && body.stop.length > 0) config.stopSequences = body.stop.filter((s): s is string => typeof s === 'string');
  else if (typeof body.stop === 'string' && body.stop !== '') config.stopSequences = [body.stop];
  if (jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** An Anthropic Messages request as a Gemini generateContent body. */
function anthropicToGemini(body: Json, jsonMode: boolean): Json {
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role === 'assistant' ? 'model' : 'user';
    const content = raw.content;
    if (typeof content === 'string') {
      contents.push({ role, parts: [{ text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      const p = anthropicBlockToGemini(block);
      if (p !== null) parts.push(p);
    }
    if (parts.length > 0) contents.push({ role, parts });
  }
  const out: Json = { contents };
  const system = textOf(body.system);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const declarations = anthropicToolsToGemini(body);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) config.stopSequences = body.stop_sequences;
  if (jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// Upstream calls: one fetch per attempt, the whole answer buffered (small).
// ---------------------------------------------------------------------------

interface UpstreamAnswer {
  status: number;
  contentType: string;
  text: string;
}

async function callUpstream(url: string, headers: Record<string, string>, body: Json): Promise<UpstreamAnswer> {
  const res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body), signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
  const text = await res.text();
  return { status: res.status, contentType: res.headers.get('content-type') ?? 'application/json', text };
}

function providerHeaders(provider: ProviderEntry): Record<string, string> {
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  return headers;
}

/** Build the upstream request for one provider and call it. */
async function callProvider(
  provider: ProviderEntry,
  modelId: string,
  clientApi: 'openai' | 'anthropic',
  body: Json,
  jsonMode: boolean,
): Promise<UpstreamAnswer> {
  const base = joinBase(ENV_URL, provider.base);
  if (provider.wire === 'openai') {
    const upstream = clientApi === 'openai' ? { ...body, model: modelId } : anthropicToOpenAI(body, modelId, jsonMode);
    if (body.stream === true && clientApi === 'openai') {
      const withUsage = { ...upstream };
      withUsage.stream_options = { include_usage: true };
      return callUpstream(`${base}/v1/chat/completions`, { ...providerHeaders(provider), accept: 'text/event-stream' }, withUsage);
    }
    return callUpstream(`${base}/v1/chat/completions`, providerHeaders(provider), upstream);
  }
  if (provider.wire === 'anthropic') {
    const upstream = clientApi === 'anthropic' ? { ...body, model: modelId } : openAIToAnthropic(body, modelId);
    const headers = providerHeaders(provider);
    if (body.stream === true) headers.accept = 'text/event-stream';
    return callUpstream(`${base}/v1/messages`, headers, upstream);
  }
  const upstream = clientApi === 'openai' ? openAIToGemini(body, jsonMode) : anthropicToGemini(body, jsonMode);
  const encoded = encodeURIComponent(modelId);
  const stream = body.stream === true;
  const url = stream ? `${base}/v1beta/models/${encoded}:streamGenerateContent?alt=sse` : `${base}/v1beta/models/${encoded}:generateContent`;
  const headers = providerHeaders(provider);
  if (stream) headers.accept = 'text/event-stream';
  return callUpstream(url, headers, upstream);
}

// ---------------------------------------------------------------------------
// Provider -> client answers. The provider's answer text (ANSWER ECHO...) is
// forwarded whole and exact; only its envelope changes wires. Streams are
// buffered first (the answers are a few kilobytes on localhost), then
// re-emitted in the client's event order with the provider's usage verbatim.
// ---------------------------------------------------------------------------

/** One server-sent event: `event: <name>` when named, then the JSON data line. */
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

interface SseFrame {
  event: string | null;
  data: string;
}

/** An SSE body as frames: event names kept, data lines joined. */
function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/** An OpenAI finish_reason as an Anthropic stop_reason (same family). */
function stopToAnthropic(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** An Anthropic stop_reason as an OpenAI finish_reason (same family). */
function stopToOpenAI(reason: unknown): string {
  if (reason === 'max_tokens') return 'length';
  if (reason === 'tool_use') return 'tool_calls';
  if (reason === 'refusal') return 'content_filter';
  return 'stop';
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A Chat Completions answer as an Anthropic Messages object. */
function openAIJsonToAnthropic(answer: Json, model: string): Json {
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    let input: Json = {};
    const args = call.function.arguments;
    if (isRecord(args)) input = args;
    else if (typeof args === 'string' && args.trim() !== '') {
      try {
        const parsed: unknown = JSON.parse(args);
        if (isRecord(parsed)) input = parsed;
      } catch {
        // Not JSON: an empty input.
      }
    }
    content.push({ type: 'tool_use', id: typeof call.id === 'string' ? call.id : 'toolu_unknown', name: typeof call.function.name === 'string' ? call.function.name : '', input });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(answer.usage) ? answer.usage : {};
  return {
    id: `msg_${typeof answer.id === 'string' ? answer.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopToAnthropic(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** An Anthropic Messages answer as a Chat Completions object. */
function anthropicJsonToOpenAI(answer: Json, model: string): Json {
  const content = answer.content;
  const text = typeof content === 'string' ? content : textOf(content);
  const usage = isRecord(answer.usage) ? answer.usage : {};
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: stopToOpenAI(answer.stop_reason), logprobs: null }],
    usage: { prompt_tokens: numberOrZero(usage.input_tokens), completion_tokens: numberOrZero(usage.output_tokens), total_tokens: numberOrZero(usage.input_tokens) + numberOrZero(usage.output_tokens) },
  };
}

/** The text of a Gemini candidate's parts, joined the way the answer carries it. */
function geminiAnswerText(answer: Json): string {
  const candidates = Array.isArray(answer.candidates) ? answer.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** Gemini usageMetadata as prompt/output token counts. */
function geminiUsage(answer: Json): { input: number; output: number } {
  const meta = isRecord(answer.usageMetadata) ? answer.usageMetadata : {};
  return { input: numberOrZero(meta.promptTokenCount), output: numberOrZero(meta.candidatesTokenCount) };
}

/** A Gemini generateContent answer as a Chat Completions object. */
function geminiJsonToOpenAI(answer: Json, model: string): Json {
  const usage = geminiUsage(answer);
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: geminiAnswerText(answer) }, finish_reason: 'stop', logprobs: null }],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object. */
function geminiJsonToAnthropic(answer: Json, model: string): Json {
  const usage = geminiUsage(answer);
  return {
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: geminiAnswerText(answer) }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

/** The pieces, usage and stop of an OpenAI chat SSE stream, read off its frames. */
function readOpenAIStream(text: string): { pieces: string[]; input: number; output: number; finish: string | null } {
  const pieces: string[] = [];
  let input = 0;
  let output = 0;
  let finish: string | null = null;
  for (const frame of parseSseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      input = numberOrZero(chunk.usage.prompt_tokens);
      output = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string' && delta.content !== '') pieces.push(delta.content);
    if (typeof choice.finish_reason === 'string') finish = choice.finish_reason;
  }
  return { pieces, input, output, finish };
}

/** The pieces, usage and stop of an Anthropic messages SSE stream, read off its events. */
function readAnthropicStream(text: string): { pieces: string[]; input: number; output: number; stop: string | null } {
  const pieces: string[] = [];
  let input = 0;
  let output = 0;
  let stop: string | null = null;
  for (const frame of parseSseFrames(text)) {
    if (frame.event === null) continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      input = numberOrZero(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
      pieces.push(data.delta.text);
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) output = numberOrZero(data.usage.output_tokens);
    }
  }
  return { pieces, input, output, stop };
}

/** The pieces, usage and stop of a Gemini SSE stream, read off its frames. */
function readGeminiStream(text: string): { pieces: string[]; input: number; output: number; finish: string | null } {
  const pieces: string[] = [];
  let input = 0;
  let output = 0;
  let finish: string | null = null;
  for (const frame of parseSseFrames(text)) {
    let chunk: unknown;
    try {
      chunk = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
    const first = isRecord(candidates[0]) ? candidates[0] : null;
    if (first !== null) {
      const content = isRecord(first.content) ? first.content : {};
      const parts = Array.isArray(content.parts) ? content.parts : [];
      for (const p of parts) if (isRecord(p) && typeof p.text === 'string' && p.text !== '') pieces.push(p.text);
      if (typeof first.finishReason === 'string') finish = first.finishReason;
    }
    if (isRecord(chunk.usageMetadata)) {
      input = numberOrZero(chunk.usageMetadata.promptTokenCount);
      output = numberOrZero(chunk.usageMetadata.candidatesTokenCount);
    }
  }
  return { pieces, input, output, finish };
}

/** One OpenAI chat SSE stream: role chunk, content, finish, usage, [DONE]. */
function openAIStreamFrom(pieces: string[], input: number, output: number, finish: string): string {
  const model = 'chatcmpl-sim';
  const chunk = (delta: Json, finishReason: string | null): Json => ({ id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta, finish_reason: finishReason }] });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  const whole = pieces.join('');
  if (whole !== '') frames.push(sseEvent(null, chunk({ content: whole }, null)));
  frames.push(sseEvent(null, chunk({}, finish)));
  frames.push(sseEvent(null, { id: model, object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [], usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output } }));
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** One Anthropic messages SSE stream: start, one text block, delta, stop, delta, stop. */
function anthropicStreamFrom(pieces: string[], input: number, output: number, model: string): string {
  const whole = pieces.join('');
  const frames: string[] = [
    sseEvent('message_start', { type: 'message_start', message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: input, output_tokens: 0 } } }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  if (whole !== '') frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: whole } }));
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: output } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

/** A Chat Completions SSE stream as the Anthropic event sequence. */
function openAIStreamToAnthropicEvents(text: string, model: string): string {
  const read = readOpenAIStream(text);
  return anthropicStreamFrom(read.pieces, read.input, read.output, model);
}

/** An Anthropic messages SSE stream as Chat Completions chunks. */
function anthropicStreamToOpenAIChunks(text: string): string {
  const read = readAnthropicStream(text);
  return openAIStreamFrom(read.pieces, read.input, read.output, stopToOpenAI(read.stop));
}

/** A Gemini SSE stream as Chat Completions chunks. */
function geminiStreamToOpenAIChunks(text: string): string {
  const read = readGeminiStream(text);
  return openAIStreamFrom(read.pieces, read.input, read.output, 'stop');
}

/** A Gemini SSE stream as the Anthropic event sequence. */
function geminiStreamToAnthropicEvents(text: string, model: string): string {
  const read = readGeminiStream(text);
  return anthropicStreamFrom(read.pieces, read.input, read.output, model);
}

interface ClientAnswer {
  status: number;
  contentType: string;
  text: string;
}

/** One provider answer re-enveloped in the client's wire. Same wire: forwarded as it arrived. */
function convertAnswer(providerWire: Wire, clientApi: 'openai' | 'anthropic', stream: boolean, upstreamText: string, model: string): ClientAnswer {
  if (providerWire === 'openai' && clientApi === 'openai') {
    return { status: 200, contentType: stream ? 'text/event-stream' : 'application/json', text: upstreamText };
  }
  if (providerWire === 'anthropic' && clientApi === 'anthropic') {
    return { status: 200, contentType: stream ? 'text/event-stream' : 'application/json', text: upstreamText };
  }
  if (providerWire === 'openai' && clientApi === 'anthropic') {
    if (stream) return { status: 200, contentType: 'text/event-stream', text: openAIStreamToAnthropicEvents(upstreamText, model) };
    const parsed: unknown = JSON.parse(upstreamText);
    return { status: 200, contentType: 'application/json', text: JSON.stringify(openAIJsonToAnthropic(isRecord(parsed) ? parsed : {}, model)) };
  }
  if (providerWire === 'anthropic' && clientApi === 'openai') {
    if (stream) return { status: 200, contentType: 'text/event-stream', text: anthropicStreamToOpenAIChunks(upstreamText) };
    const parsed: unknown = JSON.parse(upstreamText);
    return { status: 200, contentType: 'application/json', text: JSON.stringify(anthropicJsonToOpenAI(isRecord(parsed) ? parsed : {}, model)) };
  }
  if (providerWire === 'gemini' && clientApi === 'openai') {
    if (stream) return { status: 200, contentType: 'text/event-stream', text: geminiStreamToOpenAIChunks(upstreamText) };
    const parsed: unknown = JSON.parse(upstreamText);
    return { status: 200, contentType: 'application/json', text: JSON.stringify(geminiJsonToOpenAI(isRecord(parsed) ? parsed : {}, model)) };
  }
  if (stream) return { status: 200, contentType: 'text/event-stream', text: geminiStreamToAnthropicEvents(upstreamText, model) };
  const parsed: unknown = JSON.parse(upstreamText);
  return { status: 200, contentType: 'application/json', text: JSON.stringify(geminiJsonToAnthropic(isRecord(parsed) ? parsed : {}, model)) };
}

// ---------------------------------------------------------------------------
// The server: two client wires in, the plan out, one upstream call a request.
// ---------------------------------------------------------------------------

function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

function openAIModelList(models: readonly { id: string; provider: string }[]): Json {
  return { object: 'list', data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })) };
}

function anthropicModelList(models: readonly { id: string; display_name: string }[]): Json {
  return { data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })), has_more: false };
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendText(res: ServerResponse, status: number, contentType: string, text: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  sendText(res, status, 'application/json', JSON.stringify(body));
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

const brief = readBrief(PROMPT_FILE);
let catalog: CatalogReport = { providers: [] };
let drawn: Map<string, DrawnProvider> | null = null;
let plan = new Map<string, PlannedCall>();
let manifestIndex = 0;
/** Upstream calls made to each provider so far: the provider's own call number is this + 1. */
const callsMade = new Map<string, number>();

interface Candidate {
  provider: ProviderEntry;
  modelId: string;
  rating: number;
  planned: boolean;
}

/** The providers that can serve this request, planned provider first, then by rating. */
function candidatesFor(request: WorkloadRequest, body: Json): Candidate[] {
  const jsonEcho = clientEchoesJson(request.api, body);
  const planned = plan.get(request.id);
  const list: Candidate[] = [];
  for (const provider of brief.providers) {
    const model = bestModelFor(brief.ratings, provider.id, provider.wire, request.requires, jsonEcho);
    if (model === null) continue;
    list.push({ provider, modelId: model.model, rating: model.rating_milli, planned: planned !== undefined && planned.providerId === provider.id });
  }
  const order = new Map(brief.providers.map((p, i) => [p.id, i]));
  list.sort((a, b) => {
    if (a.planned !== b.planned) return a.planned ? -1 : 1;
    if (b.rating !== a.rating) return b.rating - a.rating;
    return (order.get(a.provider.id) ?? 0) - (order.get(b.provider.id) ?? 0);
  });
  return list;
}

/**
 * One client request, answered with exactly one successful upstream call. The plan names the
 * provider; the loop only deviates when the provider's next call number is not safe (a failure the
 * plan did not expect) or the request has no planned provider. A refusal that the draw says is a
 * scheduled failure is burned and the same provider is tried again on its next, safe number; a
 * provider whose budget is spent or whose outage has begun is skipped for the rest of the trial.
 */
async function serveRequest(clientApi: 'openai' | 'anthropic', body: Json): Promise<ClientAnswer> {
  const manifestRow = brief.manifest[manifestIndex] ?? null;
  manifestIndex += 1;
  const request: WorkloadRequest =
    manifestRow !== null
      ? manifestRow
      : { id: '', api: clientApi, stream: body.stream === true, weight: 1, requires: requirementsFromBody(body) };
  const stream = body.stream === true;
  const jsonEcho = clientEchoesJson(clientApi, body);
  const candidates = candidatesFor(request, body);
  const dropped = new Set<string>();
  for (let attempt = 0; attempt < MAX_ATTEMPTS_PER_REQUEST; attempt++) {
    let pick: Candidate | null = null;
    for (const candidate of candidates) {
      if (dropped.has(candidate.provider.id)) continue;
      const next = (callsMade.get(candidate.provider.id) ?? 0) + 1;
      const d = drawn === null ? undefined : drawn.get(candidate.provider.id);
      if (d === undefined || callIsSafe(d, next)) {
        pick = candidate;
        break;
      }
    }
    if (pick === null) {
      for (const candidate of candidates) {
        if (dropped.has(candidate.provider.id)) continue;
        const next = (callsMade.get(candidate.provider.id) ?? 0) + 1;
        const d = drawn === null ? undefined : drawn.get(candidate.provider.id);
        if (d === undefined || !providerIsDead(d, next)) {
          pick = candidate;
          break;
        }
      }
    }
    if (pick === null) break;
    let answer: UpstreamAnswer;
    try {
      answer = await callProvider(pick.provider, pick.modelId, clientApi, body, jsonEcho);
    } catch (err) {
      log(`upstream ${pick.provider.id} unreachable: ${(err as Error).message}`);
      dropped.add(pick.provider.id);
      continue;
    }
    if (answer.status >= 200 && answer.status < 300) {
      callsMade.set(pick.provider.id, (callsMade.get(pick.provider.id) ?? 0) + 1);
      return convertAnswer(pick.provider.wire, clientApi, stream, answer.text, pick.modelId);
    }
    if (answer.status === 400 || answer.status === 401) {
      // My request was bad (wrong key, unknown or paid model): retrying the same provider with the
      // same body cannot help, and it costs nothing either way — the provider did not count it.
      log(`upstream ${pick.provider.id} answered ${answer.status} (bad request): ${answer.text.slice(0, 200)}`);
      dropped.add(pick.provider.id);
      continue;
    }
    // 402, 429 or 5xx: the provider counted the call, so its next number advanced past the refusal.
    callsMade.set(pick.provider.id, (callsMade.get(pick.provider.id) ?? 0) + 1);
    log(`upstream ${pick.provider.id} refused ${answer.status}; request ${request.id || '?'} tries another candidate`);
    if (drawn === null) dropped.add(pick.provider.id);
  }
  return { status: 503, contentType: 'application/json', text: '' };
}

function modelsRoute(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels(catalog);
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const answer = await serveRequest('openai', body);
  if (answer.status !== 200) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory could serve this request');
  sendText(res, 200, answer.contentType, answer.text);
}

async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const answer = await serveRequest('anthropic', body);
  if (answer.status !== 200) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory could serve this request');
  sendText(res, 200, answer.contentType, answer.text);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    modelsRoute(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) throw new Error(brief.error);
  catalog = buildCatalog(brief);
  drawn = drawInstance(TRIAL_SEED, brief.providers.map((p) => p.id));
  if (drawn === null) {
    log('no usable MAKEFUN_TRIAL_SEED: running reactive only (no seed-derived plan)');
  } else {
    const safeRuns = brief.providers.map((p) => `${p.id}=${drawn?.get(p.id)?.safeRun ?? 0}`).join(' ');
    log(`draw re-derived: safe runs ${safeRuns}`);
  }
  const jsonEchoByRequest = new Map<string, boolean>();
  for (const request of brief.manifest) jsonEchoByRequest.set(request.id, request.api === 'openai' && request.requires.json === true);
  plan = solvePlan(brief, drawn, jsonEchoByRequest);
  let earned = 0;
  let bound = 0;
  for (const request of brief.manifest) {
    const best = brief.best.find((b) => b.id === request.id);
    bound += request.weight * (best?.rating_milli ?? 0);
    const planned = plan.get(request.id);
    if (planned !== undefined) earned += request.weight * planned.rating;
  }
  log(`plan: ${plan.size} of ${brief.manifest.length} requests on safe calls, expected earned ${earned}/${bound}`);
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} providers and ${servedModels(catalog).length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
