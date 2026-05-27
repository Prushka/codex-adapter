# codex-adapter

`codex-adapter` is a local proxy that lets Codex speak the OpenAI Responses API to an upstream provider that only exposes an OpenAI-compatible Chat Completions API.

It accepts Codex requests on `/v1/responses`, translates them to `/v1/chat/completions`, then translates upstream Chat Completions responses or streaming chunks back into Responses API events. The adapter always forces the configured upstream model and `reasoning_effort` so Codex cannot accidentally send provider-incompatible values.

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
  -listen 127.0.0.1:8080 \
  -provider-url http://localhost:1234/v1 \
  -model your-chat-model \
  -reasoning-effort medium \
  -api-key-env UPSTREAM_API_KEY
```

Point Codex at the adapter:

```toml
model_provider = "codex-adapter"
model = "gpt-5.5" # placeholder for model_catalog compatibility, actual model is configured in the adapter
disable_response_storage = true
model_catalog_json = "~/.codex/models.json" # replace with the latest version from codex, IMPORTANT: update context window settings so compact could work properly!

[model_providers.codex-adapter]
name = "codex-adapter"
base_url = "http://127.0.0.1:8080/v1"
wire_api = "responses"
supports_websockets = false

```

The `-provider-url` value may be an upstream base URL, a `/v1` URL, or a direct `/chat/completions` URL. The adapter normalizes it to the upstream Chat Completions endpoint.

Post setup:

1. Turn off Image Gen skill as it doesn't work as of now. This may lead the model to emit image generation tool calls.

## Endpoints

- `GET /healthz`: returns `{"ok":true}`.
- `GET /models` and `GET /v1/models`: returns a one-model list for the configured upstream model.
- `POST /responses` and `POST /v1/responses`: accepts Responses API create requests and streams Responses SSE events.
- `POST /responses/compact` and `POST /v1/responses/compact`: accepts the same request shape, sends a non-streaming upstream request, and returns `{"output":[...]}` with translated Responses output items.

Set `-disable-upstream-streaming` to make `/responses` buffer the upstream Chat Completions response and then emit the translated Responses events in one burst.

## Options

| Flag                          | Default          | Description                                                                                       |
|-------------------------------|------------------|---------------------------------------------------------------------------------------------------|
| `-listen`                     | `127.0.0.1:8080` | Local listening address for Codex requests.                                                       |
| `-provider-url`               | required         | Upstream OpenAI-compatible base URL, `/v1` URL, or direct Chat Completions URL.                   |
| `-model`                      | required         | Upstream model forced into every Chat Completions request.                                        |
| `-reasoning-effort`           | `medium`         | `reasoning_effort` forced into every upstream request.                                            |
| `-reasoning-history`          | `auto`           | Historical Responses reasoning translation: `auto`, `drop`, `reasoning-content`, or `assistant-content`. |
| `-api-key-env`                | unset            | Environment variable containing the upstream provider API key.                                    |
| `-api-key`                    | unset            | Upstream provider API key supplied directly on the command line. Prefer `-api-key-env`.           |
| `-disable-upstream-streaming` | `false`          | Buffer upstream Chat Completions responses instead of requesting SSE chunks.                       |
| `-search-provider`            | `duckduckgo`     | Local web search backend: `auto`, `duckduckgo`, `duckduckgo-lite`, `bing`, `yahoo`, or `searxng`. |
| `-search-url`                 | unset            | Backend URL for providers that need one. Required for `searxng`.                                  |
| `-debug`                      | `false`          | Write translated requests, responses, SSE events, and search activity as ordered JSON files.      |
| `-debug-dir`                  | `debug`          | Directory for debug JSON files.                                                                   |
| `-timeout`                    | `10m`            | Timeout for upstream and local search HTTP requests.                                              |

`-api-key` and `-api-key-env` are mutually exclusive. If neither is set, the adapter forwards the inbound `Authorization` header sent by Codex. If either adapter-owned key source is set, it overrides Codex's inbound authorization and is sent upstream as `Authorization: Bearer <key>`. Values that already include an authorization scheme, such as `Bearer sk-...`, are forwarded as-is.

## Local Web Search

Chat Completions has no standard hosted `web_search` tool, so the adapter exposes `web_search` to the upstream model as a synthetic function. When the upstream model makes a single `web_search` call, the adapter:

1. Emits the corresponding Responses `web_search_call` item for Codex.
2. Runs the requested local search action.
3. Appends the search result as a Chat Completions tool message.
4. Sends a follow-up Chat Completions request so Codex receives a completed turn.

Supported search actions:

- `search`: searches one `query` or multiple `queries`.
- `open_page`: fetches and extracts readable text from a URL.
- `find_in_page`: fetches a URL and returns excerpts around a pattern.

Domain filters from `domains` or `filters.allowed_domains` are applied after search results are normalized. The default backend filters obvious ad/click-tracking results, tries DuckDuckGo first, then Bing and Yahoo if the primary backend is blocked or returns no parseable results. `duckduckgo-lite` starts with DuckDuckGo Lite before using the same fallbacks. For built-in search backends, medium and high context searches also add short page excerpts from the top organic results.

The adapter applies the inbound Responses `web_search` tool config as defaults for synthetic Chat Completions tool calls. `search_context_size` is tuned for large-context models: `low` returns up to 5 concise results, `medium` returns up to 10 results and enriches the top 5 pages, and `high` returns up to 20 results and enriches the top 8 pages with larger excerpts. `open_page` and `find_in_page` also scale their extracted text budget with the same setting.

Example SearXNG setup:

```sh
go run ./cmd/codex-adapter \
  -listen 127.0.0.1:8080 \
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
- `web_search` and `image_generation` are exposed as synthetic functions because Chat Completions has no standard equivalent for Responses hosted tools. Completed `web_search_call` history is replayed from a bounded in-memory cache as synthetic assistant tool calls plus tool results when possible, and dropped instead of rendered as raw Responses JSON when the cache is unavailable. `image_generation` is marked completed only when the upstream call includes base64 image data.
- `tool_choice`, `parallel_tool_calls`, and Responses JSON schema text formats are translated to their Chat Completions equivalents.
- Nonessential Responses-only provider fields, such as `metadata`, `prompt_cache_key`, `store`, and `service_tier`, are not forwarded upstream.
- Gemini/OpenAI compatibility metadata such as `tool_calls[].extra_content.google.thought_signature` and assistant message `extra_content` is preserved on Responses items where possible and cached so follow-up requests can send it back upstream even when Codex drops unknown item fields.
- Streaming Chat Completions chunks are accumulated and emitted as Responses SSE events. A normal completion ends with `response.completed`; upstream `length` and `content_filter` finish reasons become `response.incomplete`.

The adapter is stateless across Responses turns except for bounded caches of provider compatibility `extra_content` and synthetic web-search history. Tool-call metadata is keyed by `call_id`; assistant-message metadata is keyed by message content and occurrence; web-search history is keyed by normalized action and occurrence. Codex sends the full input on this wire path, so `previous_response_id` is not required.

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
