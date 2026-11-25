import { createServer } from "node:http";
import { randomUUID } from "node:crypto";
import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { runCoworldPlayer } from "@cogweb/coworld";
import { parseJsonAction } from "@cogweb/llm";
import { runCoworldGame } from "../coworld/server.js";
import { decide as baselineDecide } from "./baseline-player.js";
import { cogherenceGame, cogherenceModule, renderPlayerMessages } from "./game.js";
import { makeLlmDecide, makeLlmTalk } from "./llm-player.js";
import { renderSpeechMessages } from "./speech.js";
import { TrainingSession } from "./training-bridge.js";
import { SUBMIT_ORDERS_FORMAT } from "../agents/llm/submit.js";

afterEach(() => {
  vi.unstubAllEnvs();
});

async function startDeterministicModelServer(
  handler: (req: { slot: string; messages: Array<{ role: string; content: string }> }) => string,
) {
  let calls = 0;
  const server = createServer((req, res) => {
    let body = "";
    req.on("data", (chunk) => {
      body += chunk;
    });
    req.on("end", () => {
      calls++;
      const parsed = JSON.parse(body);
      const slot = String(req.headers["x-coworld-player-slot"] ?? "");
      const content = handler({ slot, messages: parsed.messages });
      res.writeHead(200, {
        "content-type": "application/json",
        "X-Softmax-Llm-Call-Id": randomUUID(),
      });
      res.end(
        JSON.stringify({
          model: "deterministic/test",
          choices: [{ message: { content }, finish_reason: "stop" }],
          usage: { prompt_tokens: 10, completion_tokens: 5 },
        }),
      );
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const addr = server.address();
  if (!addr || typeof addr === "string") throw new Error("Expected TCP address");
  return {
    url: `http://127.0.0.1:${addr.port}`,
    getCalls: () => calls,
    close: () => new Promise<void>((resolve, reject) => server.close((e) => (e ? reject(e) : resolve()))),
  };
}

describe("published model player & gameplay contracts", () => {
  it("calls HTTP, parses bare and fenced JSON, and applies non-hold actions against 3 explicit baseline opponents", async () => {
    const dir = await mkdtemp(join(tmpdir(), "cogherence-model-vs-baseline-"));
    vi.stubEnv("COGAME_RESULTS_URI", join(dir, "results.json"));
    vi.stubEnv("COGAME_SAVE_TRAJECTORY_URI", join(dir, "trajectory.jsonl"));
    let orderTurns = 0;
    const model = await startDeterministicModelServer(({ messages }) => {
      const user = messages[1]!.content;
      if (user.includes("NEGOTIATION")) {
        return '```json\n{"messages":[{"to":"public","text":"Bidding for hearts."}]}\n```';
      }
      orderTurns++;
      // Bid 1 energy on the first 5 turns (using a standard outer Markdown JSON fence on turn 1), then hold.
      if (orderTurns === 1) return '```json\n{"bid":1}\n```';
      if (orderTurns <= 5) return '{"bid":1}';
      return "{}";
    });
    vi.stubEnv("COWORLD_LLM_ENDPOINT", model.url);
    vi.stubEnv("COWORLD_LLM_MODEL", "deterministic/test");
    const host = await runCoworldGame({
      host: "127.0.0.1",
      port: 0,
      config: {
        tokens: ["t0", "t1", "t2", "t3"],
        players: [{ name: "Model" }, { name: "Base1" }, { name: "Base2" }, { name: "Base3" }],
        seed: 7,
      },
    });
    try {
      const players = [
        runCoworldPlayer({
          module: cogherenceModule,
          connect: host.playerUrls[0],
          decide: makeLlmDecide(),
          talk: makeLlmTalk(),
        }),
        ...host.playerUrls.slice(1).map((connect) =>
          runCoworldPlayer({
            module: cogherenceModule,
            connect,
            decide: baselineDecide,
          }),
        ),
      ];
      const [result] = await Promise.all([host.finished, Promise.all(players)]);
      expect(model.getCalls()).toBe(200); // 100 talk + 100 orders for seat 0
      expect(result.scores[0]).toBe(5); // seat 0 won 5 hearts via HTTP model bids
      expect(result.scores.slice(1)).toEqual([0, 0, 0]);
    } finally {
      await host.close();
      await model.close();
    }
  }, 30_000);

  it("fails explicitly and preserves raw model replies instead of falling back to baseline", async () => {
    const dir = await mkdtemp(join(tmpdir(), "cogherence-fail-explicit-"));
    vi.stubEnv("COGAME_RESULTS_URI", join(dir, "results.json"));
    vi.stubEnv("COGAME_SAVE_TRAJECTORY_URI", join(dir, "trajectory.jsonl"));
    const rawBadReply = 'Sure! Here is my move: {"orders":[{"type":"bid","energy":1}]}';
    const model = await startDeterministicModelServer(({ messages }) => {
      if (messages[1]!.content.includes("NEGOTIATION")) return '{"messages":[]}';
      return rawBadReply;
    });
    vi.stubEnv("COWORLD_LLM_ENDPOINT", model.url);
    vi.stubEnv("COWORLD_LLM_MODEL", "deterministic/test");
    const host = await runCoworldGame({
      host: "127.0.0.1",
      port: 0,
      config: {
        tokens: ["t0", "t1", "t2", "t3"],
        players: [{ name: "Model" }, { name: "Base1" }, { name: "Base2" }, { name: "Base3" }],
        seed: 7,
      },
    });
    try {
      const players = [
        runCoworldPlayer({
          module: cogherenceModule,
          connect: host.playerUrls[0],
          decide: makeLlmDecide(),
          talk: makeLlmTalk(),
        }),
        ...host.playerUrls.slice(1).map((connect) =>
          runCoworldPlayer({
            module: cogherenceModule,
            connect,
            decide: baselineDecide,
          }),
        ),
      ];
      await expect(host.finished).rejects.toThrow(/failed/i);
      await host.close();
      await Promise.allSettled(players);
      const trajectory = JSON.parse(await readFile(join(dir, "trajectory.jsonl"), "utf8"));
      expect(trajectory.episode.status).toBe("failed");
      const failedDecision = trajectory.decisions.find(
        (d: { action_status: string; seat: string; observation?: { phase?: string } }) =>
          d.action_status === "failed" && d.seat === "0" && d.observation?.phase !== "talk",
      );
      expect(failedDecision).toBeDefined();
      expect(failedDecision.executed_action).toBeNull();
      expect(
        failedDecision.attempts.some(
          (a: { response: string; rejection_reason: string | null }) =>
            a.response === rawBadReply && a.rejection_reason !== null,
        ),
      ).toBe(true);
    } finally {
      await host.close();
      await model.close();
    }
  }, 15_000);

  it("fails explicitly when LLM credentials are absent instead of playing baseline", async () => {
    vi.stubEnv("COWORLD_LLM_ENDPOINT", "");
    vi.stubEnv("OPENROUTER_API_KEY", "");
    const state = cogherenceGame.newGame({ seed: "7", playerCount: 4, seatNames: ["A", "B", "C", "D"] });
    const view = cogherenceGame.redact(state, 0);
    const decide = makeLlmDecide();
    await expect(
      decide({
        view,
        signal: new AbortController().signal,
        playerSlot: 0,
        seat: 0,
        turn: 1,
        messages: [],
        reason: null,
        timeLeftMs: null,
        config: {},
        module: cogherenceModule,
        recordAttempt: () => {},
        markFallback: () => {},
      }),
    ).rejects.toThrow(/COWORLD_LLM_ENDPOINT|OPENROUTER_API_KEY/);
  });

  it("enforces strict JSON parsing (bare or outer fence only), canonical schema, and consistent prompts", () => {
    expect(parseJsonAction('{"bid":1}')).toEqual({ kind: "parsed", value: { bid: 1 } });
    expect(parseJsonAction('```json\n{"bid":1}\n```')).toEqual({ kind: "parsed", value: { bid: 1 } });
    expect(parseJsonAction('Here you go: {"bid":1}').kind).toBe("rejected");
    expect(parseJsonAction("[]").kind).toBe("rejected");

    // No tie-break contradiction between schema and rules.
    expect(SUBMIT_ORDERS_FORMAT.inputSchema.properties.bid.description).not.toContain("committed first");
    expect(SUBMIT_ORDERS_FORMAT.inputSchema.properties.bid.description).toContain("earlier seat");

    const state = cogherenceGame.newGame({ seed: "7", playerCount: 4, seatNames: ["A", "B", "C", "D"] });
    const view = cogherenceGame.redact(state, 0);
    const orderMsgs = renderPlayerMessages(view, 0, null);
    expect(orderMsgs[1]!.content).not.toContain("empty list holds");
    expect(orderMsgs[1]!.content).toContain("Example:");
    const talkMsgs = renderSpeechMessages(view, 0, []);
    expect(talkMsgs[1]!.content).toContain("Example:");

    // TrainingSession rejects invalid replies without ever synthesizing a fallback action.
    const session = new TrainingSession("7", 4);
    const talkObs = session.observation();
    if (talkObs.kind !== "decision") throw new Error("expected decision");
    expect(session.step(talkObs.decision_id, '{"messages":[{"to":"public","text":"   "}]}').kind).toBe("rejected");
    expect(session.step(talkObs.decision_id, '{"messages":[]}').kind).toBe("accepted");
    const orderObs = session.observation();
    if (orderObs.kind !== "decision") throw new Error("expected decision");
    // Rejects undocumented { orders: [] } alias and repeated invalid replies without fallback.
    for (let i = 0; i < 4; i++) {
      const res = session.step(orderObs.decision_id, '{"orders":[]}');
      expect(res.kind).toBe("rejected");
      expect(res.observation).toMatchObject({ decision_id: orderObs.decision_id });
    }
    // Accepts standard outer Markdown JSON fence.
    expect(session.step(orderObs.decision_id, '```json\n{"bid":1}\n```').kind).toBe("accepted");
  });
});
