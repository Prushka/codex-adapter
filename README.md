# codex-adapter

`codex-adapter` is a Responses-to-Chat-Completions bridge for Codex with full tool-call translation, hosted-tool emulation, reasoning replay, and Gemini/Kimi compatibility.

It lets Codex use OpenAI-compatible Chat Completions providers while preserving Codex's Responses-style workflow: standard function tools, `apply_patch`, local web search, `tool_search`, provider reasoning fields, and compatibility metadata such as Gemini thought signatures.

It accepts Codex requests on `/v1/responses`, translates them to `/v1/chat/completions`, then translates upstream Chat Completions responses or streaming chunks back into Responses API events. The adapter always forces the configured upstream model and `reasoning_effort` so Codex cannot accidentally send provider-incompatible values.

## Features

- Bridges Codex's Responses API wire format to OpenAI-compatible Chat Completions providers.
- Translates standard function tools, namespace tools, custom/freeform tools, `apply_patch`, `tool_search`, `web_search`, and `image_generation`.
- Executes hosted-style `web_search` locally with DuckDuckGo, Bing, Yahoo, or SearXNG backends, including `search`, `open_page`, and `find_in_page`.
- Handles pure web-search turns, parallel web-search turns, and mixed `web_search` plus client-tool turns while replaying search history in the correct tool-call order.
- Preserves provider compatibility metadata, including Gemini `extra_content.google.thought_signature` and Kimi `reasoning_content` history where supported.
- Supports streaming and non-streaming upstream Chat Completions responses, `/responses/compact`, debug trace output, and an optional Chat Completions and Responses load-balancer proxy.

## Requirements

- Go 1.26 or newer.
- An upstream provider with an OpenAI-compatible Chat Completions endpoint.
- A Codex provider config that uses the Responses wire API.

## Quick Start

Set the upstream API key in the environment:

```sh
export UPSTREAM_API_KEY=your-upstream-key
```

Run the adapter:

```sh
go run ./cmd/codex-adapter \
  -listen 127.0.0.1:18080 \
  -provider-url http://localhost:1234/v1 \
  -model your-chat-model \
  -reasoning-effort medium \
  -api-key-env UPSTREAM_API_KEY
```

Point Codex at the adapter:

```toml
model_provider = "codex-adapter"
model = "gpt-5" # any Codex catalog model id; the upstream model is forced by the adapter
disable_response_storage = true
model_catalog_json = "~/.codex/models.json" # keep this copied from the Codex version you run; tune context windows to match the upstream model so compact has the right budget

[model_providers.codex-adapter]
name = "codex-adapter"
base_url = "http://127.0.0.1:18080/v1"
wire_api = "responses"
supports_websockets = false

```

The `-provider-url` value may be an upstream base URL, a `/v1` URL, or a direct `/chat/completions` URL. The adapter normalizes it to the upstream Chat Completions endpoint.

Post setup:

1. Turn off Codex image-generation skills unless the upstream model can return base64 image data through the synthetic `image_generation` function. The adapter can translate the call shape, but it can only complete an `image_generation_call` when the upstream response includes image bytes.

## Endpoints

- `GET /healthz`: returns `{"ok":true}`.
- `GET /models` and `GET /v1/models`: returns a one-model list for the configured upstream model.
- `POST /responses` and `POST /v1/responses`: accepts Responses API create requests and streams Responses SSE events.
- `POST /responses/compact` and `POST /v1/responses/compact`: accepts the same request shape, sends a non-streaming upstream request, and returns `{"output":[...]}` with translated Responses output items.

Set `-disable-upstream-streaming` to make `/responses` buffer the upstream Chat Completions response and then emit the translated Responses events in one burst.

## Options

| Flag                          | Default           | Description                                                                                       |
|-------------------------------|-------------------|---------------------------------------------------------------------------------------------------|
| `-listen`                     | `127.0.0.1:18080` | Local listening address for Codex requests.                                                       |
| `-provider-url`               | required          | Upstream OpenAI-compatible base URL, `/v1` URL, or direct Chat Completions URL.                   |
| `-model`                      | required          | Upstream model forced into every Chat Completions request.                                        |
| `-reasoning-effort`           | `medium`          | `reasoning_effort` forced into every upstream request.                                            |
| `-reasoning-history`          | `auto`            | Historical Responses reasoning translation: `auto`, `drop`, `reasoning-content`, or `assistant-content`. |
| `-api-key-env`                | unset             | Environment variable containing the upstream provider API key.                                    |
| `-api-key`                    | unset             | Upstream provider API key supplied directly on the command line. Prefer `-api-key-env`.           |
| `-disable-upstream-streaming` | `false`           | Buffer upstream Chat Completions responses instead of requesting SSE chunks.                       |
| `-search-provider`            | `duckduckgo`      | Local web search backend: `auto`, `duckduckgo`, `duckduckgo-lite`, `bing`, `yahoo`, or `searxng`. |
| `-search-url`                 | unset             | Backend URL for providers that need one. Required for `searxng`.                                  |
| `-debug`                      | `false`           | Write translated requests, responses, SSE events, and search activity as ordered JSON files.      |
| `-debug-dir`                  | `debug`           | Directory for debug JSON files.                                                                   |
| `-timeout`                    | `10m`             | Timeout for upstream and local search HTTP requests.                                              |

`-api-key` and `-api-key-env` are mutually exclusive. If neither is set, the adapter forwards the inbound `Authorization` header sent by Codex. If either adapter-owned key source is set, it overrides Codex's inbound authorization and is sent upstream as `Authorization: Bearer <key>`. Values that already include an authorization scheme, such as `Bearer sk-...`, are forwarded as-is.

## Chat Completions and Responses Load Balancer

`cmd/load-balancer` is a small pass-through load balancer for OpenAI-compatible Chat Completions and Responses create requests. It forwards `POST /chat/completions`, `POST /v1/chat/completions`, `POST /responses`, and `POST /v1/responses` request bodies and headers to configured upstream providers, overriding only `Authorization`. It also serves `GET /models` and `GET /v1/models` as an OpenAI-compatible union of all configured provider models, plus `GET /models/map` and `GET /v1/models/map` as a provider-to-models mapping. `GET /providers`, `GET /v1/providers`, `GET /providers/status`, and `GET /v1/providers/status` return provider status snapshots with provider IDs, advertised models, busy counts, recent failures, last successful request time, cooldown state, and model-refresh state. `POST /refresh` and `POST /v1/refresh` reload the provider config and refresh model lists immediately. On startup it fetches each provider's `/models` list and only forwards a request to providers that advertise the requested `model`. If no configured provider supports the model, the proxy returns a JSON `model_not_available` error.

Requests are assigned from a provider pool. For each downstream request, the proxy builds one provider order by sorting matching providers by ascending `tier`, then randomizing providers within each tier. Within the current tier, the proxy prefers the next idle matching provider; if every untried provider in that tier is busy, it sends to the next provider from that tier before moving to a higher tier. If an upstream attempt fails with a request error or non-2xx HTTP response, the same request body is retried against the next untried matching provider. By default, `-attempts 5` repeats the full matching provider pool up to five times, with `-delay 1m` between full-pool attempts. The first attempt has no delay, and later full-pool attempts reuse the same provider order chosen for the request. If every configured attempt returns an HTTP failure, the last upstream error response is returned unchanged.

Repeated request failures temporarily cool down a provider. By default, `-provider-cooldown-failures 3` consecutive Chat Completions or Responses failures put that provider into cooldown for `-provider-cooldown 1m`; set either value to `0` to disable cooldown. Providers in cooldown remain visible in `/v1/providers/status`, but request routing skips them until the cooldown expires. Model-refresh failures are reported in status but do not count toward request cooldown.

Provider model lists refresh in the background every 30 minutes by default. Before each model refresh, the proxy rereads the YAML provider config so providers can be added, removed, or updated without restarting. Use `-model-refresh-interval 0` to disable background refreshes, `-model-refresh-timeout` to control the timeout for each provider's `/models` request, or call `POST /refresh` to refresh immediately. A failed startup fetch or refresh records provider status, clears that provider's model list, and skips the provider for model-routed traffic until a later refresh succeeds.

The load balancer does not translate between API formats. Use the Chat Completions routes only with upstream providers that support Chat Completions, and use the Responses routes only with upstream providers that support Responses.

```sh
go run ./cmd/load-balancer \
  -listen 127.0.0.1:18081 \
  -api-key sk-load-balancer \
  -config ./config.yaml
```

The load balancer requires `-api-key`. Downstream clients must send it as `Authorization: Bearer <key>`. This key is checked only by the load balancer and is never sent upstream; upstream requests always use the per-provider key from the selected provider config entry.

The provider config is YAML:

```yaml
providers:
  - id: p1
    url: https://provider-one.example.com/v1
    key: sk-provider-one
    tier: 0
  - id: p2
    url: https://provider-two.example.com/v1
    key: sk-provider-two
    tier: 1
```

Each `id` is only for logs and error messages, but it must be unique. The optional `tier` defaults to `0`; lower numeric tiers are tried before higher tiers, while providers with the same tier are randomized per downstream request. The `url` may be a base URL, a `/v1` URL, or a direct `/chat/completions`, `/responses`, or `/models` URL. Direct endpoint URLs are treated as siblings, so `https://provider.example/v1/responses` also implies `https://provider.example/v1/chat/completions` and `https://provider.example/v1/models`. `config.yaml` is ignored by git because it usually contains provider keys; use `config.example.yaml` as a template.

## Local Web Search

Chat Completions has no standard hosted `web_search` tool, so the adapter exposes `web_search` to the upstream model as a synthetic function.

When the upstream model emits only `web_search` calls, including parallel searches, the adapter:

1. Emits the corresponding Responses `web_search_call` item for Codex.
2. Runs the requested local search action.
3. Appends each search result as a Chat Completions tool message in call order.
4. Sends a follow-up Chat Completions request so Codex receives a completed turn.

When the upstream model emits a mixed turn, such as `web_search`, `web_search`, and `exec_command` in the same assistant message, the adapter executes and caches the web searches but does not auto-follow-up immediately. Codex receives the `web_search_call` items and client tool calls, executes the client tools, then sends the full history back. The adapter replays the cached web-search results as Chat Completions tool messages alongside the client tool outputs so the upstream provider sees one coherent parallel tool-call turn.

Supported search actions:

- `search`: searches one `query` or multiple `queries`.
- `open_page`: fetches and extracts readable text from a URL.
- `find_in_page`: fetches a URL and returns excerpts around a pattern.

Domain filters from `domains` or `filters.allowed_domains` are applied after search results are normalized. The default backend filters obvious ad/click-tracking results, tries DuckDuckGo first, then Bing and Yahoo if the primary backend is blocked or returns no parseable results. `duckduckgo-lite` starts with DuckDuckGo Lite before using the same fallbacks. For built-in search backends, medium and high context searches also add short page excerpts from the top organic results.

The adapter applies the inbound Responses `web_search` tool config as defaults for synthetic Chat Completions tool calls. `search_context_size` is tuned for large-context models: `low` returns up to 5 concise results, `medium` returns up to 10 results and enriches the top 5 pages, and `high` returns up to 20 results and enriches the top 8 pages with larger excerpts. `open_page` and `find_in_page` also scale their extracted text budget with the same setting.

Example SearXNG setup:

```sh
go run ./cmd/codex-adapter \
  -listen 127.0.0.1:18080 \
  -provider-url http://localhost:1234/v1 \
  -model your-chat-model \
  -reasoning-effort medium \
  -search-provider searxng \
  -search-url http://localhost:8081/search \
  -api-key-env UPSTREAM_API_KEY
```

## Translation Behavior

- Responses `instructions` become a system message. `developer` input messages are also sent upstream as system messages for Chat Completions compatibility.
- Responses `input` strings and message items become Chat Completions `messages`. User `input_image` content is translated to Chat Completions `image_url` parts.
- Upstream `reasoning_content`, `reasoning`, and `reasoning_delta` fields are translated to Responses reasoning output items where possible.
- Historical Responses reasoning items are not sent upstream as normal assistant text by default. `-reasoning-history auto` sends them as Chat Completions `reasoning_content` for Kimi K2 thinking/K2.6 models, adds `thinking: {"type":"enabled","keep":"all"}` for Kimi K2.6 preserved thinking, and drops them for other models. Use `-reasoning-history reasoning-content` for other compatible providers, `drop` for strict providers such as DeepSeek reasoner-style APIs, or `assistant-content` for the previous legacy behavior.
- Responses function tools become Chat Completions `function` tools.
- Namespace tools are flattened using Codex-style names and reconstructed with `namespace` on the way back. Tool names with a reserved `mcp__` prefix are renamed before sending upstream.
- Responses custom/freeform tools, including `apply_patch`, are exposed upstream as strict functions with one required string property named `input`. Returned calls are reconstructed as `custom_tool_call` items.
- `tool_search` is exposed as a synthetic function and reconstructed as a client-executed `tool_search_call`. Tool definitions discovered through `tool_search_output` are registered for later follow-up calls.
- `web_search` and `image_generation` are exposed as synthetic functions because Chat Completions has no standard equivalent for Responses hosted tools. Completed `web_search_call` history is replayed from a bounded in-memory cache as synthetic assistant tool calls plus tool results when possible. If a web-search item cannot be matched to cached execution history, it is omitted instead of being rendered as raw Responses JSON. Cached web-search replay preserves ordering with adjacent client tool calls, including mixed and parallel tool turns. `image_generation` is marked completed only when the upstream call includes base64 image data.
- `tool_choice`, `parallel_tool_calls`, and Responses JSON schema text formats are translated to their Chat Completions equivalents.
- Nonessential Responses-only provider fields, such as `metadata`, `prompt_cache_key`, `store`, and `service_tier`, are not forwarded upstream.
- Gemini/OpenAI compatibility metadata such as `tool_calls[].extra_content.google.thought_signature` and assistant message `extra_content` is preserved on Responses items where possible and cached so follow-up requests can send it back upstream even when Codex drops unknown item fields.
- Streaming Chat Completions chunks are accumulated and emitted as Responses SSE events. A normal completion ends with `response.completed`; upstream `length` and `content_filter` finish reasons become `response.incomplete`.

The adapter is stateless across Responses turns except for bounded caches of provider compatibility `extra_content` and synthetic web-search history. Tool-call metadata is keyed by `call_id`; assistant-message metadata is keyed by message content and occurrence; web-search history is keyed by normalized action and occurrence. Codex sends the full input on this wire path, so `previous_response_id` is not required.

## Provider Notes

- **Gemini OpenAI compatibility**: Gemini may return `extra_content.google.thought_signature` on assistant messages or tool calls. The adapter keeps that metadata on outbound Responses items and replays it on later Chat Completions assistant messages and assistant tool-call messages.
- **Kimi thinking models**: `-reasoning-history auto` enables `reasoning-content` history for Kimi K2 thinking/K2.6 model names. For Kimi K2.6 preserved thinking, the adapter also sends `thinking: {"type":"enabled","keep":"all"}` and keeps `reasoning_content` attached to assistant message and tool-call history.
- **Strict providers**: Use `-reasoning-history drop` for providers that reject historical reasoning fields.

## Debugging

Enable `-debug` to write ordered JSON files for inbound Responses requests, upstream Chat Completions requests and responses, outbound Responses events, and local web search activity:

```sh
go run ./cmd/codex-adapter \
  -provider-url http://localhost:1234/v1 \
  -model your-chat-model \
  -debug \
  -debug-dir ./debug
```

Upstream HTTP failures are returned to Codex as failed Responses events and logged with the upstream status, request IDs when present, content type, and a truncated response body.

## Tests

Run the test suite with:

```sh
go test ./...
```
