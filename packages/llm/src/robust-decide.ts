// The robust decide loop — the shared backbone for BOTH action decisions and
// message generation. Ask the model for one decision, parse its sampled JSON text, then `validate`
// (schema + legality). On a thrown rejection, re-prompt INCLUDING the reason and
// retry up to `maxAttempts`; on exhaustion fall back to `baseline()` so a flaky
// model never stalls a turn. Every attempt is reported via `recordAttempt` for
// the autopilot transcript surfaced in the UI.
//
// The catch here is legitimate control flow: catching a validation/parse
// rejection to re-prompt, catching a TRANSPORT failure (throttle, socket reset,
// 5xx, request timeout) to retry it the same way, falling back to baseline on
// exhaustion, and treating a terminal no-credentials transport error as an
// immediate baseline (re-prompting is futile — there is no model to reach). No
// failure escapes this function: the caller always gets a decision.
import type { ActAttempt, TextGeneration, GenerationPurpose } from "@cogweb/protocol";
import type { OpenRouterLlmClient, LlmMessage } from "./openrouter.js";

/**
 * Parse a JSON object from a model reply. Accepts bare JSON or a single
 * standard outer Markdown JSON fence (` ```json ... ``` ` / ` ``` ... ``` `).
 * Rejects surrounding prose or non-object JSON values.
 */
export function parseJsonAction(
  text: string,
): { kind: "parsed"; value: unknown } | { kind: "rejected"; reason: string } {
  const trimmed = text.trim();
  const fenced = trimmed.match(/^```(?:json)?\s*\r?\n?([\s\S]*?)\r?\n?\s*```$/i)?.[1];
  const candidate = (fenced ?? trimmed).trim();
  if (!candidate) return { kind: "rejected", reason: "empty reply" };
  try {
    const value: unknown = JSON.parse(candidate);
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
      return { kind: "rejected", reason: "reply must be a JSON object" };
    }
    return { kind: "parsed", value };
  } catch {
    return { kind: "rejected", reason: "reply is not a valid JSON object" };
  }
}

export function extractJson(text: string): unknown {
  const parsed = parseJsonAction(text);
  if (parsed.kind === "rejected") throw new Error(parsed.reason);
  return parsed.value;
}

export interface RobustDecideOpts<Decision> {
  client: OpenRouterLlmClient;
  signal: AbortSignal;
  purpose: GenerationPurpose;
  /** System prompt: rules, strategy, output-format contract. */
  system: string;
  slot?: number;
  /** Render the user turn. Re-rendered each attempt with the prior rejection
   *  reason (null on the first try) so the retry re-states the error. */
  renderUser: (rejection: string | null) => string;
  /** Validate a candidate (schema + legality); returns the typed decision, or
   *  THROWS a human-readable reason suitable for re-prompting. */
  validate: (candidate: unknown) => Decision;
  /** Record one attempt (prompt, response, rejection-or-null) for the transcript. */
  recordAttempt: (attempt: ActAttempt) => void;
  /** Max model calls before failing explicitly. Defaults to 3. */
  maxAttempts?: number;
  model?: string;
}

/** Run the retry loop and return a validated decision, or throw explicitly on failure. */
export async function robustDecide<Decision>(opts: RobustDecideOpts<Decision>): Promise<Decision> {
  const maxAttempts = Math.max(1, opts.maxAttempts ?? 3);
  let rejection: string | null = null;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    opts.signal.throwIfAborted();
    const generationId = globalThis.crypto.randomUUID();
    const prompt = opts.renderUser(rejection);
    const messages: LlmMessage[] = [{ role: "user", text: prompt }];
    let reply;
    let generation: TextGeneration | undefined;
    try {
      reply = await opts.client.complete({
        signal: opts.signal,
        purpose: opts.purpose,
        system: opts.system,
        messages,
        slot: opts.slot,
        model: opts.model,
        recordGeneration: (evidence) => {
          generation = evidence;
          opts.recordAttempt({
            generationId,
            purpose: opts.purpose,
            prompt,
            response: evidence.response,
            error: null,
            generation: evidence,
            platformCallId: evidence.platformCallId ?? null,
            providerRequestId: evidence.providerRequestId ?? null,
          });
        },
      });
    } catch (err) {
      // Preserve transport/credential failure evidence and surface the error explicitly.
      opts.recordAttempt({
        generationId,
        purpose: opts.purpose,
        prompt,
        response: generation?.response ?? "",
        error: err instanceof Error ? err.message : String(err),
        generation,
        platformCallId: generation?.platformCallId ?? null,
        providerRequestId: generation?.providerRequestId ?? null,
      });
      throw err;
    }
    opts.signal.throwIfAborted();
    const response = reply.text;

    try {
      // Parse AND validate inside the same try: a parse failure (no JSON in the
      // reply) is a rejection to re-prompt exactly like a legality rejection.
      const candidate = extractJson(reply.text);
      const decision = opts.validate(candidate);
      opts.recordAttempt({
        generationId,
        purpose: opts.purpose,
        prompt,
        response,
        error: null,
        generation: reply.generation,
        parsedAction: decision,
        platformCallId: reply.platformCallId,
        providerRequestId: reply.generation.providerRequestId ?? null,
      });
      return decision;
    } catch (err) {
      rejection = err instanceof Error ? err.message : String(err);
      opts.recordAttempt({
        generationId,
        purpose: opts.purpose,
        prompt,
        response,
        error: rejection,
        generation: reply.generation,
        platformCallId: reply.platformCallId,
        providerRequestId: reply.generation.providerRequestId ?? null,
      });
    }
  }

  throw new Error(`Model decision failed after ${maxAttempts} attempt(s): ${rejection}`);
}
