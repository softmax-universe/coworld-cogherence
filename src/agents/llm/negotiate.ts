// The grouped JSON speech action accepted during negotiation, a strict zod
// validator, and a parser to Post[].
import { z } from "zod";
import type { Post } from "../types";

export const sendMessagesSchema = z
  .object({
    messages: z
      .array(
        z
          .object({
            to: z.string(),
            text: z.string().refine((s) => s.trim().length > 0, "message text must be non-empty"),
          })
          .strict(),
      )
      .max(2)
      .optional(),
  })
  .strict();

export const SEND_MESSAGES_FORMAT = {
  name: "send_messages",
  description:
    "Send negotiation messages this turn (up to 2). Use an empty list to stay silent.",
  inputSchema: {
    type: "object",
    properties: {
      messages: {
        type: "array",
        maxItems: 2,
        items: {
          type: "object",
          properties: {
            to: {
              type: "string",
              description:
                "'public' to broadcast to everyone, or a cog id like 'cog1' for a private DM",
            },
            text: { type: "string", minLength: 1, description: "one or two sentences" },
          },
          required: ["to", "text"],
          additionalProperties: false,
        },
      },
    },
    additionalProperties: false,
  },
};

/** Parse raw tool input into Post[]; throws on malformed input. */
export function parsePosts(input: unknown): Post[] {
  const data = sendMessagesSchema.parse(input);
  return (data.messages ?? []).map((m) => ({ to: m.to, text: m.text.trim() }));
}
