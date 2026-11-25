// The GAME side of the coworld player protocol. `RemotePlayerPilot` is a
// `Pilot` (kind "remote") the run loop asks for a seat's decision; it forwards
// the seat's redacted observation to an external player over a per-slot
// websocket and awaits the reply.
//
import { WebSocket } from "ws";
import { PlayerFailureError, type Pilot, type DecideContext } from "@cogweb/core";
import {
  parsePlayerFrame,
  type InboxMessage,
  type ObservationMessage,
  type TalkLine,
  type ReplyMessage,
  type FailureMessage,
} from "./protocol";

export type TalkDelivery =
  | { kind: "accepted"; messages: TalkLine[] }
  | { kind: "rejected"; reason: string };

export interface RemotePlayerPilotOpts {
  /** Base `/player` websocket URL, e.g. "ws://host:8080/player". */
  url: string;
  /** This pilot's slot; appended as the `slot` query param. */
  slot: number;
  /** The slot's auth token; appended as the `token` query param. */
  token: string;
  /** How long to wait for one reply before treating it as a failure. */
  actTimeoutMs?: number;
  /** How long to wait for the socket to open before failing. */
  connectTimeoutMs?: number;
  /** The seat's visible inbox, attached to each observation so the player sees the
   *  table talk it's entitled to. Defaults to an empty inbox (no comms). */
  inboxFor?: (seat: number) => InboxMessage[];
  /** Sink for cheap-talk lines a player posts with its reply: the host routes them
   *  to the episode bus + spectator feed and returns the authoritative applied messages. */
  onTalk: (seat: number, turn: number, lines: TalkLine[]) => TalkDelivery;
  /** Chess clock: the INITIAL wall-clock thinking bank (ms) for this policy. The
   *  pilot tells the policy its remaining bank on every observation (`timeLeftMs`),
   *  and charges the elapsed request time. An exhausted bank fails the decision. */
  chessClockMs?: number;
  /** Fischer-increment hook: called before each decision request with the current
   *  state + seat; the returned ms are CREDITED to the seat's bank (unused time
   *  banks for future turns). The game decides what counts as "a turn" (e.g.
   *  cogtan credits at each setup/preroll request). Only meaningful with
   *  `chessClockMs` set. */
  chessClockCreditFor?: (state: unknown, seat: number) => number;
}

export class RemotePlayerPilot<State, Decision> implements Pilot<State, Decision> {
  readonly kind = "remote" as const;

  readonly #url: string;
  readonly #slot: number;
  readonly #actTimeoutMs: number;
  readonly #connectTimeoutMs: number;
  readonly #inboxFor: (seat: number) => InboxMessage[];
  readonly #onTalk: (seat: number, turn: number, lines: TalkLine[]) => TalkDelivery;
  /** Configured chess-clock budget (ms), or null for an unbounded budget. */
  readonly #clockMs: number | null;
  /** Per-turn credit hook (Fischer increment), or null for a fixed total budget. */
  readonly #creditFor: ((state: unknown, seat: number) => number) | null;
  /** Remaining thinking budget (ms); only meaningful when `#clockMs` is set. */
  #bankMs: number;

  #ws: WebSocket | null = null;
  #seq = 0;
  /** The seat + turn of the in-flight observation, so a reply's cheap-talk lines
   *  can be attributed when they arrive (the bridge is per-slot, one act at a time). */
  #cur: { seat: number; turn: number } = { seat: 0, turn: 0 };
  /** The single in-flight observation's reply resolver, keyed by id. */
  #pending: {
    id: number;
    signal: AbortSignal;
    resolve: (reply: ReplyMessage | FailureMessage) => void;
    reject: (error: Error) => void;
  } | null = null;
  constructor(opts: RemotePlayerPilotOpts) {
    const u = new URL(opts.url);
    u.searchParams.set("slot", String(opts.slot));
    u.searchParams.set("token", opts.token);
    this.#url = u.toString();
    this.#slot = opts.slot;
    this.#actTimeoutMs = opts.actTimeoutMs ?? 120_000;
    this.#connectTimeoutMs = opts.connectTimeoutMs ?? 30_000;
    this.#inboxFor = opts.inboxFor ?? (() => []);
    this.#onTalk = opts.onTalk;
    this.#clockMs = opts.chessClockMs ?? null;
    this.#creditFor = opts.chessClockCreditFor ?? null;
    this.#bankMs = opts.chessClockMs ?? 0;
  }

  async decide(ctx: DecideContext<State, Decision>): Promise<Decision> {
    ctx.signal.throwIfAborted();
    // Fischer increment: credit this decision's turn allowance BEFORE the spent
    // check, so a seat that emptied its bank still gets its per-turn time (the
    // clock model is "5s per turn, bankable", not "one prepaid pool").
    if (this.#clockMs !== null && this.#creditFor) {
      this.#bankMs += Math.max(0, this.#creditFor(ctx.state, ctx.seat));
    }
    if (this.#clockMs !== null && this.#bankMs <= 0) {
      throw new Error(`remote player slot ${this.#slot} chess clock expired`);
    }

    const ws = await this.#ensureConnected(ctx.signal);
    ctx.signal.throwIfAborted();
    const view = ctx.game.redact(ctx.state, ctx.seat);
    const turn = ctx.game.turnOf(ctx.state);
    this.#cur = { seat: ctx.seat, turn };

    const startedAt = Date.now();
    const remainingBank = (): number => Math.max(0, this.#bankMs - (Date.now() - startedAt));
    try {
      const id = this.#seq++;
      const obs: ObservationMessage = {
        type: "observation",
        id,
        seat: ctx.seat,
        turn,
        view,
        messages: this.#inboxFor(ctx.seat),
        reason: null,
        // Tell the policy its remaining budget (null = no chess clock).
        timeLeftMs: this.#clockMs !== null ? Math.round(remainingBank()) : null,
      };
      let raw: ReplyMessage | FailureMessage;
      try {
        // Cap this turn's wait at the smaller of the per-turn act-timeout and the
        // remaining bank, so one turn can never overrun the whole-episode budget.
        const waitMs =
          this.#clockMs !== null
            ? Math.min(this.#actTimeoutMs, remainingBank())
            : this.#actTimeoutMs;
        raw = await this.#request(ws, obs, waitMs, ctx.signal);
      } catch (err) {
        if (this.#clockMs !== null && remainingBank() <= 0) {
          throw new Error(`remote player slot ${this.#slot} chess clock expired`);
        }
        throw err;
      }
      if (ctx.signal.aborted) {
        for (const evidence of raw.attempts ?? []) ctx.recordAttempt(evidence);
        ctx.signal.throwIfAborted();
      }
      const response = raw.type === "reply" ? JSON.stringify(raw.decision) : "";
      if (raw.type === "failure") {
        ctx.recordSpeech(raw.speechAttempts, null, true, {
          phase: "talk",
          view,
          inbox: obs.messages,
        });
        for (const evidence of raw.attempts)
          ctx.recordAttempt({ ...evidence, error: evidence.error ?? raw.error });
        throw new PlayerFailureError(raw.error, raw.failureKind ?? "player_error", ctx.seat);
      }
      ctx.recordSpeech(raw.speechAttempts ?? [], raw.messages, raw.speechUsedFallback ?? false, {
        phase: "talk",
        view,
        inbox: obs.messages,
      });
      try {
        const decision = ctx.validate(raw.decision);
        if (raw.attempts?.length) {
          for (const evidence of raw.attempts) ctx.recordAttempt(evidence);
        } else {
          ctx.recordAttempt({
            prompt: JSON.stringify(obs),
            response: JSON.stringify(decision),
            error: null,
            parsedAction: decision,
          });
        }
        return decision;
      } catch (err) {
        const reason = err instanceof Error ? err.message : String(err);
        if (raw.attempts?.length) {
          for (const evidence of raw.attempts)
            ctx.recordAttempt({ ...evidence, error: evidence.error ?? reason });
        } else {
          ctx.recordAttempt({ prompt: JSON.stringify(obs), response, parsedAction: raw.decision, error: reason });
        }
        throw new PlayerFailureError(reason, "player_error", ctx.seat);
      }
    } finally {
      // Charge this turn's wall-clock to the policy's bank (clamped at 0).
      if (this.#clockMs !== null)
        this.#bankMs = Math.max(0, this.#bankMs - (Date.now() - startedAt));
    }
  }

  /** Open the socket once and reuse it across turns. */
  #ensureConnected(signal: AbortSignal): Promise<WebSocket> {
    signal.throwIfAborted();
    if (this.#ws && this.#ws.readyState === WebSocket.OPEN) return Promise.resolve(this.#ws);
    return new Promise<WebSocket>((resolve, reject) => {
      const ws = new WebSocket(this.#url);
      const timer = setTimeout(() => {
        signal.removeEventListener("abort", onAbort);
        ws.terminate();
        reject(new Error(`remote player connect timed out after ${this.#connectTimeoutMs}ms`));
      }, this.#connectTimeoutMs);

      const onAbort = () => {
        clearTimeout(timer);
        ws.terminate();
        reject(signal.reason);
      };
      signal.addEventListener("abort", onAbort, { once: true });
      ws.on("open", () => {
        signal.removeEventListener("abort", onAbort);
        clearTimeout(timer);
        this.#ws = ws;
        resolve(ws);
      });
      ws.on("error", (err) => {
        signal.removeEventListener("abort", onAbort);
        clearTimeout(timer);
        reject(err);
      });
      ws.on("message", (data: Buffer) => this.#onMessage(data));
      ws.on("close", () => {
        this.#ws = null;
        this.#pending?.reject(new Error(`remote player slot ${this.#slot} disconnected`));
        this.#pending = null;
      });
    });
  }

  /** Send one observation and await the player's reply (or a timeout). The wait is
   *  `timeoutMs` — the per-turn act-timeout, clamped down to the remaining chess-clock
   *  bank when a clock is set, so a single turn can't overrun the episode budget. */
  #request(
    ws: WebSocket,
    obs: ObservationMessage,
    timeoutMs: number,
    signal: AbortSignal,
  ): Promise<ReplyMessage | FailureMessage> {
    signal.throwIfAborted();
    return new Promise<ReplyMessage | FailureMessage>((resolve, reject) => {
      let timer = setTimeout(() => {
        ws.send(JSON.stringify({ type: "cancel", id: obs.id }));
        this.#pending?.reject(new Error(`remote player reply timed out after ${timeoutMs}ms`));
      }, timeoutMs);
      const onAbort = () => {
        clearTimeout(timer);
        ws.send(JSON.stringify({ type: "cancel", id: obs.id }));
        timer = setTimeout(
          () => this.#pending?.reject(new Error("Remote player cancellation did not settle")),
          1000,
        );
      };
      this.#pending = {
        id: obs.id,
        signal,
        resolve: (response) => {
          clearTimeout(timer);
          signal.removeEventListener("abort", onAbort);
          this.#pending = null;
          resolve(response);
        },
        reject: (error) => {
          clearTimeout(timer);
          signal.removeEventListener("abort", onAbort);
          this.#pending = null;
          reject(error);
        },
      };
      signal.addEventListener("abort", onAbort, { once: true });
      ws.send(JSON.stringify(obs));
    });
  }

  /** Deliver cheap talk independently of move legality and correlate the reply by ID. */
  #onMessage(data: Buffer): void {
    const parsed = parsePlayerFrame(data.toString());
    if (parsed.kind !== "message") {
      console.error(`[remote-pilot] slot ${this.#slot}: dropping ${parsed.kind} player frame`);
      return;
    }
    const msg = parsed.message;
    if (msg.type !== "artifact_complete" && this.#pending && msg.id === this.#pending.id) {
      if (msg.type === "reply" && (msg.usedFallback || msg.speechUsedFallback)) {
        const error = "Player returned a fallback instead of a policy decision";
        this.#pending.resolve({
          type: "failure",
          id: msg.id,
          error,
          attempts: msg.attempts ?? [],
          speechAttempts: (msg.speechAttempts ?? []).map(attempt => ({ ...attempt, error })),
        });
        return;
      }
      if (msg.type === "reply" && !this.#pending.signal.aborted && msg.messages.length > 0) {
        const delivery = this.#onTalk(this.#cur.seat, this.#cur.turn, msg.messages);
        if (delivery.kind === "accepted") {
          msg.messages = delivery.messages;
        } else {
          this.#pending.resolve({
            type: "failure",
            id: msg.id,
            error: delivery.reason,
            attempts: msg.attempts ?? [],
            speechAttempts: (msg.speechAttempts ?? []).map((attempt) => ({
              ...attempt,
              error: attempt.error ?? delivery.reason,
            })),
          });
          return;
        }
      }
      this.#pending.resolve(msg);
    }
  }

  /** Close the socket (call when the episode ends). */
  close(): void {
    this.#ws?.close();
    this.#ws = null;
  }
}
