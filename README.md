# free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/Krwanz/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/Krwanz/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

Free-model aggregator: a rating-routed OpenAI/Anthropic server over ten free-tier providers, with failover across all three wires.

## Run it

```sh
node main.ts --list
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=a15ed58m&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>

## How it works

`main.ts` (Node 22, standard library only) reads the trial brief
(`$MAKEFUN_PROMPT_FILE`): the ten providers, the sealed quality table and
the workload manifest. It probes each provider's catalog route in its own
wire, builds one catalog, registers its port with the env, and serves the
client's 20 requests:

- `POST /v1/chat/completions` and `POST /v1/messages`, JSON and SSE
- `GET /v1/models`, OpenAI shape for `Authorization: Bearer`, Anthropic
  shape for `x-api-key`

Routing: every free model that meets a request's requirements (tools,
vision, JSON mode, context length) is ranked by the quality table's
`rating_milli`, best first, and tried in that order until one answers 2xx.
A provider that refuses (402/429/5xx) is skipped for the rest of that
request and is not the first pick of the next one; a provider that refuses
twice in a row is treated as out of budget. A JSON-mode request from an
OpenAI client is never sent to an Anthropic-wire provider, whose wire
cannot carry `response_format`.

The wires are translated faithfully: the provider echoes the system prompt,
user text, tools, `max_tokens`, stop sequences, image and JSON mode, and
the client compares that echo with what it sent, so every translation
preserves those fields exactly, and every answer returns the provider's
text whole with the provider's usage counts and stop reason mapped
(`stop`/`end_turn`/`STOP`, `length`/`max_tokens`, `tool_calls`/`tool_use`,
`content_filter`/`refusal`).

Deterministic: no wall clock (fixed timestamps in every reply) and no
unseeded randomness, so a replay leaves the same state export.

## Practice scores

Against the kit's three practice trials, with the sealed scorer:

| trial | score | quality | served | wire | failover | catalog | calls |
|-------|-------|---------|--------|------|----------|---------|-------|
| 0 | 99 | 99 | 100 | 100 | 100 | 100 | 27 |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 | 30 |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 | 29 |

Two runs of trial 0 leave byte-identical state exports.
