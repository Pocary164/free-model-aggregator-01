# Free Model Aggregator

One zero-dependency Python 3.13 server that fronts ten free-tier model
providers behind a single OpenAI- and Anthropic-compatible API.

## What it does

- Serves `POST /v1/chat/completions` (OpenAI Chat Completions) and
  `POST /v1/messages` (Anthropic Messages), each in JSON and in
  server-sent events, plus `GET /v1/models` answered in the wire the
  caller's auth header names (`Authorization: Bearer` gets the OpenAI
  list, `x-api-key` the Anthropic one).
- At startup it reads the trial brief (`MAKEFUN_PROMPT_FILE`), fetches
  every provider's own model catalog over that provider's wire and auth
  scheme, and registers the catalog it built with the env
  (`POST $MAKEFUN_ENV_URL/register`).
- For each request it picks the highest-rated **free** model that meets
  the request's requirements (tools, images, JSON mode, context length),
  translates the request into that provider's wire (OpenAI, Anthropic or
  Gemini), and translates the answer back into the client's wire:
  streaming event order, stop reasons, usage, tool calls and their
  results, the system prompt, `max_tokens`, stop sequences and images
  are carried faithfully.
- When a provider refuses (402/429, a failure on its schedule, or an
  outage) it fails over to the next-best legal model, tracks each
  provider's refusals, and never calls a paid model.
- It polls `GET $MAKEFUN_ENV_URL/work` and exits 0 when the workload is
  done. Every output is a deterministic function of the brief, the
  requests and the providers' answers, so a replayed trial leaves a
  byte-identical record.

## Usage

```
python3 main.py          # run the aggregator server (trial mode)
python3 main.py --list   # print the catalog document and exit
```

Run outside a trial (no `MAKEFUN_PROMPT_FILE` / `MAKEFUN_ENV_URL`),
`--list` prints `{"providers":[]}`. Inside a trial it prints exactly the
catalog document the server registers.

## Layout

Everything lives in `main.py` (standard library only): brief parsing,
provider catalog fetching, the canonical request/answer translation
layer for the three provider wires and the two client wires, the
quality-ordered router with budget-aware failover, and the HTTP server.
