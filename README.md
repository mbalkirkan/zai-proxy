# z.ai OpenAI Proxy

OpenAI-compatible `/v1` proxy for `chat.z.ai`. The server accepts both `POST /v1/chat/completions` and `POST /v1/responses` requests and forwards them to z.ai using your logged-in token.

## What it supports

- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /zai/chat`
- `DELETE /zai/chat/{chat_id}`
- `GET /v1/models`
- `GET /healthz`
- Non-stream responses only
- Text-only messages
- `system` / `developer` instructions are folded into the next user turn for z.ai compatibility
- Automatic retry on temporary upstream capacity errors
- Automatic chat cleanup after each response
- Built-in request spacing to reduce IP-ban risk from burst traffic

## Setup

1. Create a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Copy `.env.example` to `.env` and set `ZAI_TOKEN`.
4. Start the server.

Optional tuning:

- `ZAI_DELETE_CHAT_AFTER_RESPONSE=true` keeps temporary chats from piling up
- `ZAI_MIN_REQUEST_INTERVAL_MS=2000` spaces outbound z.ai requests to avoid bursts
- `PROXY_BEARER_TOKEN=...` enables bearer protection for incoming requests

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
- `ZAI_MODEL`
- `ZAI_FE_VERSION`
- `PROXY_BEARER_TOKEN`
- `ZAI_DELETE_CHAT_AFTER_RESPONSE=true`
- `ZAI_MIN_REQUEST_INTERVAL_MS=2000`
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

Use the base URL:

```text
http://YOUR_HOST:8000/v1
```

If you set `PROXY_BEARER_TOKEN` or `ZAI_PROXY_API_KEY`, send it as:

```text
Authorization: Bearer YOUR_PROXY_TOKEN
```

## Limits

- This proxy currently creates a fresh z.ai chat for each request.
- That temporary chat is deleted after the response by default.
- `/zai/chat` is the exception: it keeps the chat until you delete it.
- It does not support `stream=true` on either endpoint.
- It only supports text content parts.

## Cleanup existing chats

```bash
python scripts/cleanup_chats.py
```

To keep the newest 10 chats:

```bash
python scripts/cleanup_chats.py --keep 10
```
