# Deployment

This engine runs entirely on **GitHub Actions** — no server, no VPS, no Docker required.
Every job is a scheduled workflow that installs Python, runs one task, and exits.

## Architecture

```
GitHub Actions (free tier)
  ├── edge-calculator.yml   every 5 min  → poll Polymarket + alert Telegram
  ├── news-poller.yml       every 10 min → RSS + Reddit + Mistral classify
  ├── match-model.yml       every 15 min → Dixon-Coles predictions
  ├── daily-squads.yml      every hour   → Sofascore + FBref player data
  ├── tournament-sim.yml    daily 3am    → Monte Carlo sim + top scorer
  └── heartbeat.yml         daily 8am    → Telegram ping to confirm alive

Supabase (free tier) ← all jobs read/write here
Vercel (free tier)   ← dashboard reads from Supabase
Telegram bot         ← receives edge alerts
```

## GitHub Actions free tier limits

- 2,000 minutes/month on private repos
- Unlimited on public repos

Our jobs use roughly:
- edge-calculator: ~1 min × 288 runs/day = ~288 min/day
- news-poller: ~2 min × 144 runs/day = ~288 min/day
- match-model: ~2 min × 96 runs/day = ~192 min/day
- daily-squads: ~3 min × 24 runs/day = ~72 min/day
- tournament-sim: ~10 min × 1/day = ~10 min/day
- heartbeat: ~1 min × 1/day = ~1 min/day

**Total: ~850 min/day — exceeds free private repo limit.**

✅ **Solution: make the repo public.** The code has no secrets (all secrets are in
GitHub Secrets, not the code). A public repo gets unlimited Actions minutes free.

## Step-by-step setup

### 1. Make the repo public (required for unlimited Actions minutes)

- Visit your repo on GitHub
- Settings → General → scroll to "Danger Zone"
- Click **Change repository visibility** → Public → confirm

### 2. Add GitHub Secrets

Visit: `https://github.com/Graypigeon0/polymarket-wc-predictor/settings/secrets/actions`

Click **New repository secret** for each:

| Secret name | Where to get it |
|---|---|
| `SUPABASE_URL` | Supabase → Project Settings → API |
| `SUPABASE_SERVICE_KEY` | Supabase → Project Settings → API (secret key) |
| `MISTRAL_API_KEY` | console.mistral.ai → API Keys |
| `TELEGRAM_BOT_TOKEN` | @BotFather on Telegram |
| `TELEGRAM_CHAT_ID` | From getUpdates call (yours is 7963168807) |
| `FOOTBALL_DATA_API_KEY` | football-data.org (free registration) |

### 3. Run the Supabase schema migration

In Supabase dashboard → SQL Editor → paste contents of
`backend/db/migrations/001_initial_schema.sql` → click Run.

### 4. Enable the workflows

Once you push to GitHub, go to the **Actions** tab in your repo.
GitHub may ask you to enable workflows — click **Enable**.

Each workflow runs on its schedule automatically. To test one immediately:
- Click the workflow name
- Click **Run workflow** → **Run workflow**

### 5. Set up Vercel dashboard

- Visit vercel.com → New Project → import this repo
- Root directory: `frontend`
- Add env vars:
  - `NEXT_PUBLIC_SUPABASE_URL`
  - `NEXT_PUBLIC_SUPABASE_ANON_KEY`
- Deploy → install to phone home screen

## Monitoring

- **Daily heartbeat:** 8am UTC Telegram message confirms everything is alive
- **Actions tab:** `https://github.com/Graypigeon0/polymarket-wc-predictor/actions`
  shows every run with logs. Green = good. Red = something to investigate.
- **Supabase Table Editor:** check `edges`, `news_events`, `match_predictions`
  are getting rows as tournament progresses

## Updating the code

```bash
git add .
git commit -m "your change"
git push
```

Workflows automatically use the latest code on the next scheduled run.
No restart, no SSH, no rebuild needed.
