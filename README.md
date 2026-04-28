# minimax-proxy

Multi-provider Codex proxy that routes OpenAI Responses natively and translates MiniMax Chat Completions into the OpenAI API surface — so codex consumers talk to one endpoint regardless of which model actually runs.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness + configured providers |
| `GET` | `/v1/models` | Model catalog (MiniMax models only) |
| `POST` | `/v1/responses` | OpenAI Responses API — routes to MiniMax or OpenAI |
| `POST` | `/v1/chat/completions` | OpenAI Chat Completions — routes to MiniMax or OpenAI |
| `GET` | `/cop` | GitHub API-aware content proxy (GET) |
| `POST` | `/cop` | GitHub API-aware content proxy (POST) |

## Routing logic

1. Model ID is normalized and checked against explicit provider map
2. Falls back to substring match (`minimax` → MiniMax, prefix list → OpenAI)
3. Falls back to configured `DEFAULT_PROVIDER`

## Features

- **Streaming** — SSE streaming for both `/v1/responses` and `/v1/chat/completions`
- **Response store** — captures conversation history; `previous_response_id` chains are locally resolved across provider boundaries
- **Structured logging** — `tracing` with `TraceLayer`, per-request `request_id`, byte/chunk counts on every stream event

### MiniMax-only features

These features apply when routing to MiniMax:

- **web_fetch tool loop** — conversations containing URLs get `web_fetch` tool injection and an iterative resolve-loop (up to `MAX_FETCH_LOOPS`) that executes tool calls and re-sends until all URLs are resolved or the loop limit is hit
- **Circuit breaker** — detects consecutive tool-call-only responses, injects a stop nudge to prevent infinite loops, and strips tools entirely after a high-threshold pass
- **reasoning_split** — MiniMax's reasoning content is split into the response output

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MINIMAX_API_KEY` | — | MiniMax API key (required if no OpenAI key) |
| `MINIMAX_BASE_URL` | `https://api.minimax.io/v1` | MiniMax upstream base |
| `MINIMAX_MODELS` | `MiniMax-M2.7` | Available MiniMax models (CSV) |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI upstream base |
| `OPENAI_MODEL_PREFIXES` | `gpt-,o1,o3,o4,codex-,chatgpt-` | OpenAI model prefixes |
| `DEFAULT_PROVIDER` | auto | Preferred provider when model is ambiguous |
| `PROXY_PORT` | `4000` | Listen port |
| `GITHUB_TOKEN` | `gh auth token` | GitHub token for `/cop` proxy (falls back to CLI) |
| `RUST_LOG` | `minimax_proxy=info,tower_http=warn` | Log level filter |

## Building

```bash
cargo build --release -p minimax-proxy
```

## Running

```bash
PROXY_PORT=4000 MINIMAX_API_KEY=... cargo run --release -p minimax-proxy
```

## Architecture

```
Client
  │
  ▼
axum router + TraceLayer (request_id span)
  │
  ├─► health_handler        → inline
  ├─► models_handler         → inline (MiniMax model catalog)
  ├─► cop_get/post_handler  → web_fetch with GitHub token injection
  │
  ├─► responses_handler
  │     ├─ openai → forward_openai_responses (passthrough + store)
  │     └─ minimax → handle_minimax_responses
  │                     ├─ web_fetch loop (if URLs detected)
  │                     ├─ circuit breaker
  │                     ├─ streaming → handle_streaming_response (pipe + store)
  │                     └─ non-streaming → chat_completion_to_response
  │
  └─► chat_completions_handler
        ├─ openai → forward_openai_chat_completions (passthrough)
        └─ minimax → handle_minimax_chat_completions
                        ├─ web_fetch loop
                        └─ passthrough / SSE pipe
```