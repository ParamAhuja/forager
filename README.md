# Signal Terminal

An autonomous, multi-agent research terminal. It ingests live Hacker News items, separates signal from noise, enriches promising projects with GitHub metrics, drafts investment-style memos, and offers a guarded chat interface plus a sandboxed QA agent — all behind a FastAPI backend with a single-page frontend.

## Architecture

| File | Role |
|------|------|
| `server.py` | FastAPI app: pipeline orchestration, SSE chat, researcher applications (SQLite), serves the frontend |
| `api_gateway.py` | BFF secrets + Groq proxy. Only place API keys are read and `Groq()` clients are built; per-key rotation on rate limits |
| `agent1_scout.py` | **Agent 1 – Scout**: ingests HN items, classifies SIGNAL vs NOISE |
| `agent2_github_quant.py` | **Agent 2 – Quant**: GitHub enrichment and quantitative scoring |
| `agent3_partner.py` | **Agent 3 – Partner**: builds context and drafts the memo with conviction signals |
| `agent5_qa.py` | **Agent 5 – QA Engineer**: ReAct loop in an [E2B](https://e2b.dev) sandbox that clones and debugs a repo |
| `index.html` | Frontend (served at `/`) |

Agent 4 (chat) lives in `server.py` and is protected by a Llama Guard semantic firewall.

## Setup

Requires Python 3.10+.

```bash
git clone <your-repo-url>
cd signal-terminal

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # then fill in your keys
python server.py
```

Open http://localhost:8000.

## Configuration

| Variable | Required | Purpose |
|----------|----------|---------|
| `GROQ_API_KEY` or `GROQ_API_KEYS` | Yes | Groq LLM access (comma-separated list enables a rotating pool) |
| `GITHUB_TOKEN` | No | Higher GitHub API rate limits |
| `E2B_API_KEY` | For Agent 5 | E2B sandbox access |

Never commit `.env` — it is git-ignored.

## API

- `GET /` — frontend
- `POST /api/pipeline/run` — manually trigger the pipeline
- `GET /api/pipeline/state` — latest results, countdown, rate-limit state
- `GET /api/status` — key pool health and GitHub rate limit
- `POST /api/chat/stream` — Agent 4 chat (SSE)

Agent scripts can also be run standalone for validation, e.g. `python agent2_github_quant.py`.
