# codex-adapter

`codex-adapter` is a local OpenAI Responses API proxy for connecting Codex to an OpenAI-compatible provider that only exposes Chat Completions.

It listens for Codex `/v1/responses` requests, translates them to `/v1/chat/completions`, and translates upstream Chat Completions responses or streaming chunks back into Responses SSE events. The configured model and reasoning effort are forced into every upstream request.

## Usage

```sh
go run . \
  -listen 127.0.0.1:8080 \
  -provider-url http://localhost:1234/v1 \
  -model your-chat-model \
  -reasoning-effort medium \
  -api-key-env UPSTREAM_API_KEY
```

Then point Codex at `http://127.0.0.1:8080/v1` as its provider base URL.

```toml
model_provider = "codex-adapter"

[model_providers.codex-adapter]
name = "codex-adapter"
base_url = "http://127.0.0.1:8080/v1"
wire_api = "responses"
supports_websockets = false
```

Options:

- `-listen`: local listening address for Responses API requests.
- `-provider-url`: upstream OpenAI-compatible provider base URL, `/v1` URL, or direct `/chat/completions` URL.
- `-model`: upstream Chat Completions model forced into every request.
- `-reasoning-effort`: `reasoning_effort` forced into every upstream request.
- `-api-key-env`: environment variable containing the upstream provider API key. When set, the adapter overwrites Codex's inbound `Authorization` header before forwarding upstream.
- `-api-key`: upstream provider API key supplied directly on the command line. Prefer `-api-key-env` for shell history/process-list hygiene. Mutually exclusive with `-api-key-env`.
- `-debug`: writes ordered JSON debug files for inbound requests, upstream requests/responses, and outbound Responses events.
- `-debug-dir`: debug output directory, default `debug`.
- `-timeout`: upstream request timeout, default `10m`.

If neither `-api-key-env` nor `-api-key` is set, the adapter forwards the `Authorization` header that Codex sends to it. This keeps existing Codex `env_key` provider configs working. If an adapter-owned API key is configured, it always takes precedence over Codex's inbound key and is sent upstream as `Authorization: Bearer <key>`. Values that already include an authorization scheme, for example `Bearer sk-...`, are sent as-is.

## Translation Notes

- `instructions` becomes a system message for broad Chat Completions compatibility.
- Responses `input` items become chat `messages`, including prior function/custom tool calls and tool outputs.
- Responses function tools become Chat Completions `function` tools.
- Responses namespace tools are flattened with Codex's code-mode naming rule and reconstructed with `namespace` on the way back.
- Responses custom/freeform tools such as `apply_patch` are exposed upstream as a function taking one required string property, `input`, and reconstructed as `custom_tool_call`.
- `tool_search` is exposed as a synthetic chat function and reconstructed as a client-executed `tool_search_call`.
- `web_search` and `image_generation` are exposed as synthetic functions because Chat Completions has no standard equivalent for Responses hosted tools. The adapter reconstructs the corresponding Responses items if the upstream model calls them.
- `web_search` falls back to a local search-and-fetch follow-up when the upstream provider stops at a chat-completions tool call, so Codex still receives a completed Responses turn.
- Gemini/OpenAI compatibility metadata such as `tool_calls[].extra_content.google.thought_signature` is preserved on Responses tool-call items and cached by `call_id` so follow-up tool-result requests can send it back upstream.
- The adapter itself stays stateless across Responses turns. Codex sends the full input for each turn on this wire path, so `previous_response_id` is not required here.
- Streaming chat chunks are accumulated and emitted as Responses SSE events ending with `response.completed`, which Codex requires.

The implementation follows the OpenAI API reference for Responses create, Chat Completions create, and Chat Completions streaming.
