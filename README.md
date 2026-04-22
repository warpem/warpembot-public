# Warp Bot

Automated support agent for the [Warp](https://github.com/warpem/warp) cryo-EM software ecosystem. Monitors the Warp Google Group mailing list and GitHub issues on `warpem/*` repositories, drafts replies using Claude, and optionally sends them after human review.

## How it works

1. **Triage** -- checks for new emails (via AgentMail) and GitHub issues since the last run.
2. **Draft** -- invokes Claude Code (`claude -p`) for each item. Claude reads the conversation, searches a RAG index of past threads and source code, and writes a JSON draft.
3. **Consolidate** -- if multiple drafts target the same thread, a second Claude pass merges them.
4. **Review** -- drafts are printed for human review. Edit or delete files in `drafts/` before confirming.
5. **Send** -- approved drafts are sent via AgentMail (email) and the GitHub API (issue comments / new issues).

Incoming messages are indexed into a local RAG store so future runs can find similar past discussions.

## Prerequisites

- Python 3.11+
- [Conda](https://docs.conda.io/en/latest/) (any snake species compatible with your setup)
- [Ollama](https://ollama.com/) (used for local embedding generation)
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) (`claude` must be on your PATH)
- A GitHub personal access token with issue read/write on `warpem/warp`
- An [AgentMail](https://agentmail.to/) account with an inbox configured for the bot

## Setup

1. Clone the repo and its reference repositories:

```bash
git clone https://github.com/warpem/warpembot-public.git
cd warpembot-public
bash setup.sh
```

`setup.sh` clones the five reference repos into `repos/`, creates a conda environment (`warpembot`), and installs Python dependencies.

2. Set environment variables. Add these to your shell profile or a `.env` file that you source before running:

```bash
# GitHub -- personal access token with repo/issues scope on warpem/warp
export GITHUB_PAT="github_pat_..."

# AgentMail
export AGENTMAIL_API_KEY="..."
export AGENTMAIL_INBOX_ID="..."          # inbox ID, e.g. warpbot@agentmail.to
export AGENTMAIL_BOT_ADDRESS="..."       # the bot's email address
export AGENTMAIL_ESCALATION_TARGET="..." # email to forward escalations to (optional)
```

3. Pull the Ollama embedding model (happens automatically on first run, but you can do it ahead of time):

```bash
ollama pull qwen3-embedding:8b
```

## Usage

```bash
# Full run with human review (default)
bash run.sh

# Full run, send immediately without review
python run.py --no-review

# Skip triage/drafting, just review and send existing drafts
python run.py --just-send
```

On each run the bot:
- Pulls the latest code in `repos/`
- Starts Ollama if not already running
- Rebuilds code search indexes
- Triages new messages
- Generates drafts and waits for review

## Project structure

```
run.py                 Main harness: triage, draft, review, send
run.sh                 Shell wrapper (activates conda env)
setup.sh               First-time setup: clone repos, create env
github_client.py       GitHub API client (issues, comments)
agentmail_client.py    AgentMail SDK client (email threads)
rag_server.py          MCP server exposing RAG search to Claude
rag_index.py           Bulk indexing: mbox import, code index rebuild
rag_common.py          Shared RAG utilities and embedding helpers
state.py               Persistent state helpers
state.json             Processed message IDs
fetch_ccpem.py         Scraper for CCP-EM mailing list archives
CLAUDE.md              System prompt / instructions for Claude
.mcp.json              MCP server config (RAG server)
threads/               Cached message JSON files (RAG corpus)
repos/                 Cloned reference repositories (gitignored)
embeddings.npz         Thread embedding index
code_index_*.npz       Per-repo code embedding indexes
code_meta_*.json       Per-repo code metadata
```

## License

MIT
