# polymarket-wc-predictor

Hybrid statistical + LLM prediction engine for World Cup 2026 Polymarket markets.

Produces calibrated probabilities for match winners, exact scores, tournament outright, top scorer, and golden boot. Compares model probabilities against live Polymarket prices and pushes any positive-edge opportunities to a Telegram bot. Backend runs 24/7 on a free-tier cloud VM; consumed via a mobile-installable PWA dashboard.

See [`docs/BUILD_PLAN.md`](docs/BUILD_PLAN.md) for the full architecture write-up.

## Architecture at a glance

```
data sources → ingestion → postgres → modeling → edge calc → telegram + dashboard
```

- **Base model:** Dixon-Coles bivariate Poisson with recency + competition weights
- **Context layer:** Mistral AI classifies news → adjusts team ratings within hard caps
- **Tournament sim:** Monte Carlo (10k runs) for outright + stage markets
- **Top scorer:** Player-level Poisson + simulated minutes/path
- **Outputs:** Probabilities + edge vs. live Polymarket prices

## Stack

| Layer | Tool |
|---|---|
| Compute | Oracle Cloud Free Tier ARM VM |
| Database | Supabase (Postgres) |
| LLM | Mistral AI (mistral-small-latest), EU-based |
| Dashboard | Next.js PWA on Vercel |
| Alerts | Telegram bot |
| Scheduling | APScheduler + GitHub Actions |

## Setup

### Prerequisites

- Python 3.11+
- Node.js 20+
- A Supabase project (free tier)
- API keys: Gemini, Telegram bot token
- Free accounts: football-data.org, optional Groq

### 1. Clone and install

```bash
git clone https://github.com/Graypigeon0/polymarket-wc-predictor.git
cd polymarket-wc-predictor

# Backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# Frontend
cd frontend
npm install
cd ..
```

### 2. Configure environment

```bash
cp .env.example .env
# fill in the values
```

Required env vars:

- `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` — from your Supabase project settings
- `MISTRAL_API_KEY` — from console.mistral.ai (free tier)
- `TELEGRAM_BOT_TOKEN` — from `@BotFather` on Telegram
- `TELEGRAM_CHAT_ID` — your personal chat ID (send any message to your bot, then visit `https://api.telegram.org/bot<TOKEN>/getUpdates`)
- `FOOTBALL_DATA_API_KEY` — free tier at football-data.org
- `GEMINI_API_KEY` — optional fallback LLM (not available in all regions)

### 3. Run database migrations

```bash
# Apply schema in Supabase SQL editor, OR via psql:
psql "$SUPABASE_DB_URL" -f backend/db/migrations/001_initial_schema.sql
```

### 4. Run the backend locally

```bash
python -m backend.main
```

### 5. Run the dashboard locally

```bash
cd frontend
npm run dev
# open http://localhost:3000
```

## Deployment

- **Backend:** Build the Docker image, push to your Oracle VM. See `Dockerfile` + `docs/DEPLOY.md`.
- **Dashboard:** Connect this repo to Vercel; it auto-deploys on push to `main`.
- **Scheduled tasks:** GitHub Actions runs the news poller every 5 min (`.github/workflows/news-poller.yml`).

## Project structure

```
backend/
  main.py             # APScheduler driver, top-level orchestration
  config.py           # env + settings
  db/                 # Postgres client + schema migrations
  ingestion/          # Sofascore, FBref, football-data, Polymarket, RSS, Reddit
  models/             # Dixon-Coles, squad, LLM context, tournament sim, top scorer
  edges/              # Edge calculator vs. Polymarket
  alerts/             # Telegram bot
  tests/

frontend/
  app/                # Next.js App Router pages
    page.tsx          # edges dashboard (home)
    matches/[id]/     # per-match detail
    teams/[id]/       # per-team detail
  components/         # shared UI
  lib/                # supabase client, helpers
  public/             # PWA manifest, icons, service worker

notebooks/            # backtest + calibration notebooks
docs/                 # build plan + deploy notes
.github/workflows/    # CI + scheduled news polling
```

## Status

🚧 **Scaffold stage.** All modules are stubbed with clear `TODO` markers. See `docs/BUILD_PLAN.md` for the 3-week timeline.

## License

MIT — see `LICENSE`.
