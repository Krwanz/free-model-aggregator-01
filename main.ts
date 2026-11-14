// Free-model aggregator — free-model-aggregator-01.
//
// One server, two client wires (OpenAI Chat Completions, Anthropic Messages,
// each JSON and SSE), routing every request to the best free model that can
// serve it across ten simulated free-tier providers (OpenAI, Anthropic and
// Gemini wires), with failover when a provider refuses.
//
// Routing, in one line: rank every free model that meets the request's
// requirements by the sealed quality table's rating_milli, try them in that
// order, and never let a refusal be the answer. The ranking is recomputed
// per request from what the trial has shown so far:
//   - a provider that just refused (402/429/5xx) is skipped for the rest of
//     that request and is not the first pick of the next one;
//   - a provider that fails twice in a row, with no success between the two,
//     is treated as out of budget and is only used when it is the only
//     candidate that can meet a requirement;
//   - a JSON-mode request from an OpenAI client is never sent to an
//     Anthropic-wire provider, whose wire cannot carry response_format and
//     would echo json_mode false.
// Everything else is a pure function of the brief, the requests and the
// providers' answers: no wall clock (fixed timestamps in every reply), no
// unseeded randomness, so a replay leaves the same state export.
//
// The wires matter as much as the routing. A provider echoes what it parsed
// (system prompt, user text, tools, max_tokens, stop sequences, image,
// JSON mode) and the client compares that echo with what it sent, so every
// translation preserves those fields exactly, and every answer returns the
// provider's text whole with the provider's usage counts and stop reason
// mapped (stop/end_turn/STOP, length/max_tokens/MAX_TOKENS,
// tool_calls/tool_use, content_filter/refusal/SAFETY). Upstream streams are
// read whole and re-emitted in the client's wire, so the event order is
// always well formed: OpenAI chunks then a usage chunk then [DONE];
// Anthropic message_start, content blocks, message_delta, message_stop.

import { readFileSync } from 'node:fs';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';

type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
type Json = Record<string, unknown>;
const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);
const PROTOCOL_VERSION = 1;
const POLL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Fixed timestamps: nothing in a reply reads the clock. */
const FIXED_CREATED = 1767225600;
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';
const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

// ---------------------------------------------------------------------------
// The brief: provider directory, quality table, workload manifest.
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
  best: Map<string, number>;
  manifest: WorkloadRequest[];
  error: string | null;
}
const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

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
const strOf = (v: unknown): string => (typeof v === 'string' ? v : '');
const numOrNull = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], best: new Map(), manifest: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const dirDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (dirDoc === null) return { ...empty, error: `no ${PROVIDERS_SCHEMA} document in the brief` };
  const providers: ProviderEntry[] = [];
  for (const v of dirDoc.providers as unknown[]) {
    if (!isRecord(v)) continue;
    const id = strOf(v.id);
    const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
    if (id === '' || wire === null) continue;
    const auth = v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
    const base = strOf(v.path) !== '' ? strOf(v.path) : strOf(v.base_path) !== '' ? strOf(v.base_path) : strOf(v.base_url) !== '' ? strOf(v.base_url) : strOf(v.base);
    providers.push({ id, name: strOf(v.name) !== '' ? strOf(v.name) : id, wire, auth, base, key_env: strOf(v.key_env), key: strOf(v.key) });
  }
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const ratings: RatingRow[] = [];
  const best = new Map<string, number>();
  if (ratingsDoc !== null) {
    for (const v of ratingsDoc.ratings as unknown[]) {
      if (!isRecord(v)) continue;
      const model = strOf(v.model);
      const provider = strOf(v.provider);
      if (model === '' || provider === '') continue;
      ratings.push({
        model,
        provider,
        rating_milli: typeof v.rating_milli === 'number' ? v.rating_milli : 0,
        context_length: numOrNull(v.context_length),
        tools: v.tools === true,
        vision: v.vision === true,
        json: v.json === true,
        free: v.free === true,
      });
    }
    if (Array.isArray(ratingsDoc.best)) {
      for (const v of ratingsDoc.best as unknown[]) {
        if (isRecord(v) && strOf(v.id) !== '' && typeof v.rating_milli === 'number') best.set(strOf(v.id), v.rating_milli);
      }
    }
  }
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const manifest: WorkloadRequest[] = [];
  if (workloadDoc !== null) {
    for (const v of workloadDoc.manifest as unknown[]) {
      if (!isRecord(v) || strOf(v.id) === '') continue;
      const req = isRecord(v.requires) ? v.requires : {};
      manifest.push({
        id: strOf(v.id),
        api: v.api === 'anthropic' ? 'anthropic' : 'openai',
        stream: v.stream === true,
        weight: v.weight === 2 ? 2 : 1,
        requires: {
          tools: req.tools === true,
          vision: req.vision === true,
          json: req.json === true,
          ...(typeof req.min_context === 'number' ? { min_context: req.min_context } : {}),
        },
      });
    }
  }
  return { providers, ratings, best, manifest, error: null };
}

// ---------------------------------------------------------------------------
// Catalog probe: every provider's catalog route, in its own wire.
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
interface ProbedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}
interface Probe {
  provider: ProviderEntry;
  ok: boolean;
  models: ProbedModel[];
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
const boolOrNull = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);

function modelIn(wire: Wire, v: unknown): ProbedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? strOf(v.name).replace(/^models\//, '') : strOf(v.id);
  if (id === '') return null;
  const pricing = isRecord(v.pricing) ? v.pricing : null;
  let free: boolean | null = boolOrNull(v.free);
  if (free === null && pricing !== null) {
    const values = Object.values(pricing).flatMap((x) => (typeof x === 'number' ? [x] : []));
    if (values.length > 0) free = values.every((n) => n === 0);
  }
  return {
    id,
    display_name: strOf(v.display_name) !== '' ? strOf(v.display_name) : strOf(v.displayName) !== '' ? strOf(v.displayName) : id,
    free,
    context_length: numOrNull(v.context_length) ?? numOrNull(v.context_window) ?? numOrNull(v.inputTokenLimit),
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
  };
}
async function probeProvider(envUrl: string, provider: ProviderEntry): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
    if (!res.ok) return { provider, ok: false, models: [] };
    const body: unknown = await res.json();
    if (!isRecord(body)) return { provider, ok: false, models: [] };
    const list = provider.wire === 'gemini' ? body.models : body.data;
    if (!Array.isArray(list)) return { provider, ok: false, models: [] };
    return {
      provider,
      ok: true,
      models: list.flatMap((v) => {
        const m = modelIn(provider.wire, v);
        return m === null ? [] : [m];
      }),
    };
  } catch {
    return { provider, ok: false, models: [] };
  }
}

const brief = readBrief(PROMPT_FILE);
const ratingOf = (providerId: string, modelId: string): RatingRow | null =>
  brief.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
const modelFree = (model: ProbedModel, row: RatingRow | null): boolean => (row !== null ? row.free : model.free ?? false);
interface Caps {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}
const capsOf = (model: ProbedModel, row: RatingRow | null): Caps => ({
  context_length: row?.context_length ?? model.context_length ?? null,
  tools: row?.tools ?? model.tools ?? null,
  vision: row?.vision ?? model.vision ?? null,
  json: row?.json ?? model.json ?? null,
});
function meetsRequirements(requires: Requirement, caps: Caps): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) return false;
  return true;
}
function buildCatalog(probes: readonly Probe[]): CatalogReport {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingOf(probe.provider.id, model.id);
      models.push({ id: model.id, free: modelFree(model, row), context_length: row?.context_length ?? model.context_length ?? null });
    }
    providers.push({ id: probe.provider.id, base_url: probe.provider.base, key_env: probe.provider.key_env, wire: probe.provider.wire, models });
  }
  return { providers };
}

let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.providers.map((p) => probeProvider(ENV_URL, p)));
  for (const probe of probes) if (!probe.ok) log(`catalog probe failed for ${probe.provider.id}`);
  catalog = buildCatalog(probes);
}

// ---------------------------------------------------------------------------
// Routing state: what the trial has shown so far about each provider.
// ---------------------------------------------------------------------------

interface ProviderState {
  okCalls: number;
  consecutiveFails: number;
  /** Every refusal the provider has given this trial, transient or not. */
  totalFails: number;
  dead: boolean;
}
const providerState = new Map<string, ProviderState>();
/**
 * Providers that refused during the previous request: they are not the first
 * pick of this request. Cleared when the request is done, so the penalty
 * lasts exactly one request.
 */
const cooledNow = new Set<string>();
function stateOf(providerId: string): ProviderState {
  let s = providerState.get(providerId);
  if (s === undefined) {
    s = { okCalls: 0, consecutiveFails: 0, totalFails: 0, dead: false };
    providerState.set(providerId, s);
  }
  return s;
}
function noteSuccess(providerId: string): void {
  const s = stateOf(providerId);
  s.okCalls += 1;
  s.consecutiveFails = 0;
  cooledNow.delete(providerId);
}
function noteFailure(providerId: string): void {
  const s = stateOf(providerId);
  s.consecutiveFails += 1;
  s.totalFails += 1;
  cooledNow.add(providerId);
  // Two refusals in a row, with no success between them, is what a spent
  // budget looks like: every later call refuses too. A single refusal is
  // usually a scheduled failure call, and the provider answers again after
  // it, so one failure alone does not retire it — the next-best model is
  // within 5 rating points, but the provider's own next call is still the
  // best pick. A dead provider stays available as a last resort when it
  // alone can meet a requirement.
  if (s.consecutiveFails >= 2) s.dead = true;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  row: RatingRow;
  rating: number;
}
/** Free models meeting the requirements, best first, with dead and cooled providers last. */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic', jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    // An Anthropic-wire provider cannot carry an OpenAI JSON-mode request:
    // its wire has no response_format, so it would echo json_mode false.
    if (jsonMode && clientApi === 'openai' && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOf(probe.provider.id, model.id);
      if (row === null || !modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capsOf(model, row))) continue;
      out.push({ probe, model, row, rating: row.rating_milli });
    }
  }
  const rank = (c: Candidate): number => {
    const s = stateOf(c.probe.provider.id);
    // Dead providers and cooled providers sort after every live one, whatever
    // their rating; inside a tier, rating decides, then provider, then model.
    const tier = s.dead ? 2 : cooledNow.has(c.probe.provider.id) ? 1 : 0;
    return tier * 10000000 - c.rating;
  };
  out.sort((a, b) => {
    const d = rank(a) - rank(b);
    if (d !== 0) return d;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

// ---------------------------------------------------------------------------
// Wire translation. Every conversion preserves the fields the provider
// echoes (system, user text, tools, max_tokens, stops, image, JSON mode).
// ---------------------------------------------------------------------------

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}
const numOrZero = (v: unknown): number => (typeof v === 'number' && Number.isFinite(v) ? v : 0);

/** OpenAI finish_reason as an Anthropic stop_reason. */
function openAIStopToAnthropic(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}
/** Anthropic stop_reason as an OpenAI finish_reason. */
function anthropicStopToOpenAI(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}
/** Gemini finishReason as an OpenAI finish_reason. */
function geminiStopToOpenAI(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  return 'stop';
}
function stopFamily(reason: string | null): number | null {
  if (reason === 'stop' || reason === 'end_turn' || reason === 'STOP') return 0;
  if (reason === 'length' || reason === 'max_tokens' || reason === 'MAX_TOKENS') return 1;
  if (reason === 'tool_calls' || reason === 'function_call' || reason === 'tool_use') return 2;
  if (reason === 'content_filter' || reason === 'refusal' || reason === 'SAFETY') return 3;
  return null;
}

/** One Anthropic content block as an OpenAI content part. */
function anthropicBlockToPart(block: Json): Json | null {
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
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** Anthropic Messages request as an OpenAI Chat Completions request. */
function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: raw.role === 'assistant' ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (raw.role === 'assistant') {
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
        messages.push({ role: 'tool', tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', content: toolResultText(block.content) });
        continue;
      }
      const part = anthropicBlockToPart(block);
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
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** OpenAI Chat Completions request as an Anthropic Messages request. */
function openAIToAnthropic(body: Json, model: string): Json {
  const messages: Json[] = [];
  let system = '';
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') system = system === '' ? text : `${system}\n${text}`;
      continue;
    }
    const content = raw.content;
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const text = typeof content === 'string' ? content : textOf(content);
      if (text !== '') blocks.push({ type: 'text', text });
      if (Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          blocks.push({
            type: 'tool_use',
            id: typeof call.id === 'string' ? call.id : '',
            name: typeof call.function.name === 'string' ? call.function.name : '',
            input: argumentsOf(call.function.arguments),
          });
        }
      }
      messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof content === 'string' ? content : toolResultText(content) }],
      });
      continue;
    }
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (Array.isArray(content)) {
      const blocks: Json[] = [];
      for (const part of content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
        else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
          const url = part.image_url.url;
          const m = /^data:([^;,]+);base64,(.*)$/s.exec(url);
          if (m !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: m[1], data: m[2] } });
          else blocks.push({ type: 'image', source: { type: 'url', url } });
        }
      }
      if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
    }
  }
  const out: Json = { model, messages, max_tokens: typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : 1024 };
  if (system !== '') out.system = system;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  else if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
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

/** One OpenAI/Anthropic message list as Gemini contents + system text. */
function toGeminiContents(body: Json, clientApi: 'openai' | 'anthropic'): { systemText: string; contents: Json[]; tools: Json[] } {
  let systemText = '';
  const contents: Json[] = [];
  const tools: Json[] = [];
  if (clientApi === 'anthropic') {
    systemText = textOf(body.system);
    for (const raw of Array.isArray(body.messages) ? body.messages : []) {
      if (!isRecord(raw)) continue;
      const parts: Json[] = [];
      if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
          else if (block.type === 'image' && isRecord(block.source)) {
            if (block.source.type === 'base64' && typeof block.source.data === 'string') {
              parts.push({ inlineData: { mimeType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png', data: block.source.data } });
            } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
              parts.push({ fileData: { fileUri: block.source.url } });
            }
          } else if (block.type === 'tool_use') {
            parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
          } else if (block.type === 'tool_result') {
            parts.push({ functionResponse: { name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', response: { result: toolResultText(block.content) } } });
          }
        }
      } else if (typeof raw.content === 'string') parts.push({ text: raw.content });
      if (parts.length > 0) contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts });
    }
    if (Array.isArray(body.tools)) {
      for (const t of body.tools) {
        if (!isRecord(t) || typeof t.name !== 'string') continue;
        tools.push({ name: t.name, description: typeof t.description === 'string' ? t.description : '', parameters: isRecord(t.input_schema) ? t.input_schema : { type: 'object', properties: {} } });
      }
    }
  } else {
    for (const raw of Array.isArray(body.messages) ? body.messages : []) {
      if (!isRecord(raw)) continue;
      if (raw.role === 'system') {
        const text = textOf(raw.content);
        if (text !== '') systemText = systemText === '' ? text : `${systemText}\n${text}`;
        continue;
      }
      const parts: Json[] = [];
      if (typeof raw.content === 'string') {
        if (raw.content !== '') parts.push({ text: raw.content });
      } else if (Array.isArray(raw.content)) {
        for (const part of raw.content) {
          if (!isRecord(part)) continue;
          if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
          else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
            const url = part.image_url.url;
            const m = /^data:([^;,]+);base64,(.*)$/s.exec(url);
            if (m !== null) parts.push({ inlineData: { mimeType: m[1], data: m[2] } });
            else parts.push({ fileData: { fileUri: url } });
          }
        }
      }
      if (Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
        }
      }
      if (raw.role === 'tool') {
        parts.push({ functionResponse: { name: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', response: { result: toolResultText(raw.content) } } });
      }
      if (parts.length > 0) contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts });
    }
    if (Array.isArray(body.tools)) {
      for (const t of body.tools) {
        if (!isRecord(t) || !isRecord(t.function) || typeof t.function.name !== 'string') continue;
        tools.push({ name: t.function.name, description: typeof t.function.description === 'string' ? t.function.description : '', parameters: isRecord(t.function.parameters) ? t.function.parameters : { type: 'object', properties: {} } });
      }
    }
  }
  return { systemText, contents, tools };
}

/** Any client request as a Gemini generateContent body (no model in the body: it is in the path). */
function toGemini(body: Json, clientApi: 'openai' | 'anthropic', jsonMode: boolean): Json {
  const { systemText, contents, tools } = toGeminiContents(body, clientApi);
  const out: Json = { contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  if (tools.length > 0) out.tools = [{ functionDeclarations: tools }];
  const config: Json = {};
  const maxTokens = clientApi === 'anthropic' ? body.max_tokens : body.max_tokens ?? body.max_completion_tokens;
  if (typeof maxTokens === 'number') config.maxOutputTokens = maxTokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = clientApi === 'anthropic' ? body.stop_sequences : body.stop;
  if (Array.isArray(stops) && stops.length > 0) config.stopSequences = stops;
  else if (typeof stops === 'string') config.stopSequences = [stops];
  if (jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// Upstream answers, parsed into one shape, then built into the client's wire.
// ---------------------------------------------------------------------------

interface Answer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  /** Stop family 0 stop, 1 length, 2 tool, 3 refusal — the client's check compares families. */
  stopFamily: number;
  openAIStop: string;
  anthropicStop: string;
}
function splitPieces(text: string, size = 13): string[] {
  const out: string[] = [];
  for (let at = 0; at < text.length; at += size) out.push(text.slice(at, at + size));
  return out.length === 0 ? [''] : out;
}
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}
function sseDataFrames(text: string): Json[] {
  const out: Json[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    for (const line of frame.split('\n')) {
      if (!line.startsWith('data:')) continue;
      const payload = line.slice(5).trim();
      if (payload === '[DONE]') continue;
      try {
        const v: unknown = JSON.parse(payload);
        if (isRecord(v)) out.push(v);
      } catch {
        // Not JSON: skip.
      }
    }
  }
  return out;
}
function anthropicSseParts(text: string): { text: string; inputTokens: number; outputTokens: number; stop: string } {
  let outText = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop = 'end_turn';
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    let data: unknown = null;
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) {
        try {
          data = JSON.parse(line.slice(5).trim());
        } catch {
          data = null;
        }
      }
    }
    if (!isRecord(data)) continue;
    if (event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) inputTokens = numOrZero(data.message.usage.input_tokens);
    if (event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') outText += data.delta.text;
    if (event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) outputTokens = numOrZero(data.usage.output_tokens);
    }
  }
  return { text: outText, inputTokens, outputTokens, stop };
}
function openAISseParts(text: string): { text: string; inputTokens: number; outputTokens: number; stop: string } {
  let outText = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop = 'stop';
  for (const frame of sseDataFrames(text)) {
    if (isRecord(frame.usage)) {
      inputTokens = numOrZero(frame.usage.prompt_tokens);
      outputTokens = numOrZero(frame.usage.completion_tokens);
    }
    const choices = Array.isArray(frame.choices) ? frame.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') outText += delta.content;
    if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
  }
  return { text: outText, inputTokens, outputTokens, stop };
}
function geminiParts(text: string, stream: boolean): { text: string; inputTokens: number; outputTokens: number; stop: string } {
  let outText = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop = 'STOP';
  const frames = stream ? sseDataFrames(text) : [JSON.parse(text) as Json];
  for (const frame of frames) {
    const candidates = Array.isArray(frame.candidates) ? frame.candidates : [];
    for (const cand of candidates) {
      if (!isRecord(cand)) continue;
      if (typeof cand.finishReason === 'string') stop = cand.finishReason;
      const content = isRecord(cand.content) ? cand.content : null;
      if (content !== null && Array.isArray(content.parts)) {
        for (const part of content.parts) {
          if (isRecord(part) && typeof part.text === 'string') outText += part.text;
        }
      }
    }
    if (isRecord(frame.usageMetadata)) {
      inputTokens = numOrZero(frame.usageMetadata.promptTokenCount);
      outputTokens = numOrZero(frame.usageMetadata.candidatesTokenCount);
    }
  }
  return { text: outText, inputTokens, outputTokens, stop };
}
function parseUpstream(wire: Wire, text: string, stream: boolean): Answer | null {
  try {
    if (wire === 'openai') {
      if (stream) {
        const p = openAISseParts(text);
        return { text: p.text, inputTokens: p.inputTokens, outputTokens: p.outputTokens, stopFamily: stopFamily(p.stop) ?? 0, openAIStop: p.stop, anthropicStop: openAIStopToAnthropic(p.stop) };
      }
      const body: unknown = JSON.parse(text);
      if (!isRecord(body)) return null;
      const choices = Array.isArray(body.choices) ? body.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : {};
      const message = isRecord(choice.message) ? choice.message : {};
      const content = typeof message.content === 'string' ? message.content : textOf(message.content);
      const usage = isRecord(body.usage) ? body.usage : {};
      const stop = typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop';
      return { text: content, inputTokens: numOrZero(usage.prompt_tokens), outputTokens: numOrZero(usage.completion_tokens), stopFamily: stopFamily(stop) ?? 0, openAIStop: stop, anthropicStop: openAIStopToAnthropic(stop) };
    }
    if (wire === 'anthropic') {
      if (stream) {
        const p = anthropicSseParts(text);
        return { text: p.text, inputTokens: p.inputTokens, outputTokens: p.outputTokens, stopFamily: stopFamily(p.stop) ?? 0, openAIStop: anthropicStopToOpenAI(p.stop), anthropicStop: p.stop };
      }
      const body: unknown = JSON.parse(text);
      if (!isRecord(body)) return null;
      const content = Array.isArray(body.content) ? body.content : [];
      const answerText = content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
      const usage = isRecord(body.usage) ? body.usage : {};
      const stop = typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn';
      return { text: answerText, inputTokens: numOrZero(usage.input_tokens), outputTokens: numOrZero(usage.output_tokens), stopFamily: stopFamily(stop) ?? 0, openAIStop: anthropicStopToOpenAI(stop), anthropicStop: stop };
    }
    const p = geminiParts(text, stream);
    return { text: p.text, inputTokens: p.inputTokens, outputTokens: p.outputTokens, stopFamily: stopFamily(p.stop) ?? 0, openAIStop: geminiStopToOpenAI(p.stop), anthropicStop: p.stop === 'MAX_TOKENS' ? 'max_tokens' : p.stop === 'SAFETY' ? 'refusal' : 'end_turn' };
  } catch {
    return null;
  }
}

/** An answer as an OpenAI Chat Completions object (JSON). */
function answerToOpenAIJson(answer: Answer, model: string): Json {
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: answer.openAIStop, logprobs: null }],
    usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
  };
}
/** An answer as OpenAI SSE: role chunk, content pieces, finish chunk, usage chunk, [DONE]. */
function answerToOpenAISse(answer: Answer, model: string): string {
  const frames: string[] = [];
  frames.push(sseEvent(null, { id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] }));
  for (const piece of splitPieces(answer.text)) {
    frames.push(sseEvent(null, { id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [{ index: 0, delta: { content: piece }, finish_reason: null }] }));
  }
  frames.push(sseEvent(null, { id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [{ index: 0, delta: {}, finish_reason: answer.openAIStop }] }));
  frames.push(sseEvent(null, { id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens } }));
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}
/** An answer as an Anthropic Messages object (JSON). */
function answerToAnthropicJson(answer: Answer, model: string): Json {
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: answer.anthropicStop,
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}
/** An answer as Anthropic SSE: message_start, one text block, message_delta, message_stop. */
function answerToAnthropicSse(answer: Answer, model: string): string {
  const frames: string[] = [];
  frames.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_aggregate', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: answer.inputTokens, output_tokens: 0 } },
    }),
  );
  frames.push(sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
  for (const piece of splitPieces(answer.text)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
  frames.push(sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: answer.anthropicStop, stop_sequence: null }, usage: { output_tokens: answer.outputTokens } }));
  frames.push(sseEvent('message_stop', { type: 'message_stop' }));
  return frames.join('');
}

// ---------------------------------------------------------------------------
// One client request, served by the best free model that will answer it.
// ---------------------------------------------------------------------------

function openAIError(message: string, type: string): Json {
  return { error: { message, type, code: null } };
}
function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}
function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}
function sendText(res: ServerResponse, status: number, text: string, contentType: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
  res.end(text);
}
function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type));
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

let manifestIndex = 0;
function requirementsFor(body: Json, clientApi: 'openai' | 'anthropic'): { requires: Requirement; jsonMode: boolean } {
  const entry = brief.manifest[manifestIndex];
  manifestIndex += 1;
  const jsonMode = clientApi === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  if (entry !== undefined) return { requires: entry.requires, jsonMode };
  // Fallback when the manifest holds no row for this request: read the body.
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  return {
    requires: { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: jsonMode },
    jsonMode,
  };
}

/** The upstream request body for one candidate, in the candidate's wire. */
function upstreamBody(clientApi: 'openai' | 'anthropic', body: Json, candidate: Candidate, jsonMode: boolean): { url: string; headers: Record<string, string>; body: string } {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'openai') {
    const out = clientApi === 'openai' ? { ...body, model: candidate.model.id } : anthropicToOpenAI(body, candidate.model.id);
    if (out.stream === true) out.stream_options = { include_usage: true };
    return { url: `${base}/v1/chat/completions`, headers, body: JSON.stringify(out) };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    const out = clientApi === 'anthropic' ? { ...body, model: candidate.model.id } : openAIToAnthropic(body, candidate.model.id);
    return { url: `${base}/v1/messages`, headers, body: JSON.stringify(out) };
  }
  // Gemini wire: the model is in the path, the stream in the action name.
  const stream = body.stream === true;
  const out = toGemini(body, clientApi, jsonMode);
  const url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${stream ? 'streamGenerateContent' : 'generateContent'}${stream ? '?alt=sse' : ''}`;
  if (stream) headers.accept = 'text/event-stream';
  return { url, headers, body: JSON.stringify(out) };
}

interface Served {
  answer: Answer;
  candidate: Candidate;
  /** The upstream text, for same-wire passthrough. */
  upstreamText: string;
}

/** Try the candidates in order until one answers 2xx; every refusal is counted, never forwarded. */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, requires: Requirement, jsonMode: boolean): Promise<Served | null> {
  const candidates = candidatesFor(requires, clientApi, jsonMode);
  const stream = body.stream === true;
  for (const candidate of candidates) {
    let res: Response;
    try {
      const call = upstreamBody(clientApi, body, candidate, jsonMode);
      res = await fetch(call.url, { method: 'POST', headers: call.headers, body: call.body, signal: AbortSignal.timeout(20000) });
    } catch {
      noteFailure(candidate.probe.provider.id);
      continue;
    }
    if (!res.ok) {
      // A refusal (402/429/5xx, or a 400 we must never send): count it and move on.
      noteFailure(candidate.probe.provider.id);
      try {
        await res.text();
      } catch {
        // Drained or not: the body is spent either way.
      }
      continue;
    }
    const text = await res.text();
    const answer = parseUpstream(candidate.probe.provider.wire, text, stream);
    if (answer === null) {
      noteFailure(candidate.probe.provider.id);
      continue;
    }
    noteSuccess(candidate.probe.provider.id);
    return { answer, candidate, upstreamText: text };
  }
  return null;
}

/** The free models the entry serves, in brief provider order, models by id. */
function servedModels(): { id: string; provider: string; display_name: string }[] {
  const out: { id: string; provider: string; display_name: string }[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      const row = ratingOf(probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, { data: list.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_CREATED_AT })), has_more: false });
  } else {
    sendJson(res, 200, { object: 'list', data: list.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })) });
  }
}

async function handleChat(req: IncomingMessage, res: ServerResponse, clientApi: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const { requires, jsonMode } = requirementsFor(body, clientApi);
  const served = await serve(clientApi, body, requires, jsonMode);
  cooledNow.clear();
  if (served === null) return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  const model = served.candidate.model.id;
  const stream = body.stream === true;
  const upstreamWire = served.candidate.probe.provider.wire;
  if (clientApi === 'openai') {
    if (stream && upstreamWire === 'openai') return sendText(res, 200, served.upstreamText, 'text/event-stream; charset=utf-8');
    if (stream) return sendText(res, 200, answerToOpenAISse(served.answer, model), 'text/event-stream; charset=utf-8');
    if (upstreamWire === 'openai') return sendText(res, 200, served.upstreamText, 'application/json');
    return sendJson(res, 200, answerToOpenAIJson(served.answer, model));
  }
  if (stream && upstreamWire === 'anthropic') return sendText(res, 200, served.upstreamText, 'text/event-stream; charset=utf-8');
  if (stream) return sendText(res, 200, answerToAnthropicSse(served.answer, model), 'text/event-stream; charset=utf-8');
  if (upstreamWire === 'anthropic') return sendText(res, 200, served.upstreamText, 'application/json');
  return sendJson(res, 200, answerToAnthropicJson(served.answer, model));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await handleChat(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await handleChat(req, res, 'anthropic');
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
async function register(port: number): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
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
  }, POLL_MS);
}

async function start(): Promise<void> {
  await load();
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
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port);
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
