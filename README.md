# z.ai, DeepSeek, and Copilot OpenAI Proxy

OpenAI-compatible proxy for `chat.z.ai`, `chat.deepseek.com`, and GitHub Copilot Chat. z.ai uses `/v1`; DeepSeek uses `/deepseek/v1`; Copilot uses `/copilot/v1`.

## What it supports

- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /zai/chat`
- `DELETE /zai/chat/{chat_id}`
- `GET /v1/models`
- `POST /deepseek/v1/chat/completions`
- `POST /deepseek/v1/responses`
- `GET /deepseek/v1/models`
- `POST /copilot/v1/chat/completions`
- `POST /copilot/v1/responses`
- `GET /copilot/v1/models`
- `GET /healthz`
- `GET /deepseek/healthz`
- `GET /copilot/healthz`
- Non-stream responses only
- Text-only messages
- `system` / `developer` instructions are folded into the next user turn for z.ai compatibility
- DeepSeek and Copilot use a rendered single prompt for message history because their web endpoints accept one prompt per turn
- Automatic retry on temporary upstream capacity errors
- Automatic chat cleanup after each response
- Built-in request spacing to reduce IP-ban risk from burst traffic

## Setup

1. Create a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Copy `.env.example` to `.env` and set at least one of `ZAI_TOKEN`, `DEEPSEEK_TOKEN`, or `COPILOT_TOKEN`.
4. Start the server.

Optional tuning:

- `ZAI_DELETE_CHAT_AFTER_RESPONSE=true` keeps temporary chats from piling up
- `ZAI_MIN_REQUEST_INTERVAL_MS=2000` spaces full z.ai completion requests without slowing internal z.ai calls
- `PROXY_BEARER_TOKEN=...` enables bearer protection for incoming requests
- `DEEPSEEK_TOKEN=...` is the DeepSeek `localStorage.userToken.value`
- `DEEPSEEK_COOKIE=...` is optional, but may be required if DeepSeek/Cloudflare rejects server-side requests
- `DEEPSEEK_MODEL=default` maps to Instant; `expert` maps to Expert
- `DEEPSEEK_MIN_REQUEST_INTERVAL_MS=500` spaces full completion requests without slowing internal DeepSeek calls
- `COPILOT_TOKEN=...` is the GitHub Copilot `GitHub-Bearer` token copied from the browser request header
- `COPILOT_MODEL=gemini-3.1-pro-preview` defaults to the currently selected Copilot web model
- `COPILOT_MIN_REQUEST_INTERVAL_MS=500` spaces full Copilot completion requests without slowing internal Copilot calls

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn zai_proxy.main:app --host 0.0.0.0 --port 8000
```

## Docker

Build and run directly:

```bash
docker build -t zai-proxy .
docker run --env-file .env -p 8000:8000 --name zai-proxy --restart unless-stopped zai-proxy
```

Or with Compose:

```bash
docker compose up -d --build
```

## Coolify

Recommended option: deploy with the included `Dockerfile`.

Set these environment variables in Coolify:

- `ZAI_TOKEN`
- `DEEPSEEK_TOKEN`
- `COPILOT_TOKEN`
- `DEEPSEEK_COOKIE` if DeepSeek returns auth/challenge errors
- `ZAI_MODEL`
- `DEEPSEEK_MODEL=default`
- `COPILOT_MODEL=gemini-3.1-pro-preview`
- `ZAI_FE_VERSION`
- `PROXY_BEARER_TOKEN`
- `ZAI_DELETE_CHAT_AFTER_RESPONSE=true`
- `ZAI_MIN_REQUEST_INTERVAL_MS=2000`
- `DEEPSEEK_DELETE_CHAT_AFTER_RESPONSE=true`
- `DEEPSEEK_MIN_REQUEST_INTERVAL_MS=500`
- `COPILOT_DELETE_THREAD_AFTER_RESPONSE=true`
- `COPILOT_MIN_REQUEST_INTERVAL_MS=500`
- `DEEPSEEK_THINKING_ENABLED=true`
- `DEEPSEEK_SEARCH_ENABLED=false`
- `ZAI_ENABLE_THINKING=false`
- `ZAI_AUTO_WEB_SEARCH=false`

Keep container port as `8000`.

If you prefer Compose deployment, `docker-compose.yml` is also included.

## Example request

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Authorization: Bearer YOUR_PROXY_TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "GLM-5-Turbo",
    "messages": [
      {"role": "system", "content": "Be concise."},
      {"role": "user", "content": "Write one sentence about Istanbul."}
    ]
  }'
```

## DeepSeek example request

```bash
curl http://localhost:8000/deepseek/v1/chat/completions \
  -H 'Authorization: Bearer YOUR_PROXY_TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-chat",
    "messages": [
      {"role": "system", "content": "Be concise."},
      {"role": "user", "content": "Write one sentence about Istanbul."}
    ]
  }'
```

For Expert mode, use `"model": "deepseek-reasoner"` or `"model": "expert"`.

## Copilot example request

```bash
curl http://localhost:8000/copilot/v1/chat/completions \
  -H 'Authorization: Bearer YOUR_PROXY_TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemini-3.1-pro-preview",
    "messages": [
      {"role": "system", "content": "Be concise."},
      {"role": "user", "content": "Write one sentence about Istanbul."}
    ]
  }'
```

Available Copilot model ids include `gemini-3.1-pro-preview`, `gpt-5.2-codex`, `gpt-5.4-mini`, `gpt-5-mini`, `grok-code-fast-1`, `claude-haiku-4.5`, `gemini-3-flash-preview`, `gemini-2.5-pro`, `gpt-5.2`, `gpt-4.1`, and `gpt-4o`.

## Example `responses` request

```bash
curl http://localhost:8000/v1/responses \
  -H 'Authorization: Bearer YOUR_PROXY_TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "GLM-5-Turbo",
    "input": "Write one sentence about Istanbul."
  }'
```

## Stateful z.ai chat

Use `/zai/chat` when you want to keep a z.ai chat alive and continue it later.
This endpoint is not OpenAI-compatible on purpose.

Create a persistent chat:

```bash
curl http://localhost:8000/zai/chat \
  -H 'Authorization: Bearer YOUR_PROXY_TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "GLM-5-Turbo",
    "system": "Be concise.",
    "prompt": "Remember this codeword: blue-lake. Reply OK."
  }'
```

Continue the same chat:

```bash
curl http://localhost:8000/zai/chat \
  -H 'Authorization: Bearer YOUR_PROXY_TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{
    "chat_id": "CHAT_ID_FROM_PREVIOUS_RESPONSE",
    "prompt": "What codeword did I give you?"
  }'
```

The response includes `chat_id`. This chat is not deleted automatically.

Delete a persistent chat when you are done:

```bash
curl -X DELETE http://localhost:8000/zai/chat/CHAT_ID_FROM_PREVIOUS_RESPONSE \
  -H 'Authorization: Bearer YOUR_PROXY_TOKEN'
```

## n8n

Use the z.ai base URL:

```text
http://YOUR_HOST:8000/v1
```

Use the DeepSeek base URL:

```text
http://YOUR_HOST:8000/deepseek/v1
```

Use the Copilot base URL:

```text
http://YOUR_HOST:8000/copilot/v1
```

If you set `PROXY_BEARER_TOKEN` or `ZAI_PROXY_API_KEY`, send it as:

```text
Authorization: Bearer YOUR_PROXY_TOKEN
```

## Limits

- This proxy currently creates a fresh z.ai chat / DeepSeek chat / Copilot thread for each OpenAI-compatible request.
- z.ai, DeepSeek, and Copilot temporary chats are deleted after the response by default.
- `/zai/chat` is the exception: it keeps the chat until you delete it.
- It does not support `stream=true` on these endpoints.
- It only supports text content parts.
- DeepSeek and Copilot web auth is browser-session based and may require refreshing their token/cookie values.

## Cleanup existing chats

```bash
python scripts/cleanup_chats.py
```

To keep the newest 10 chats:

```bash
python scripts/cleanup_chats.py --keep 10
```
