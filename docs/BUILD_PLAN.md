# World Cup 2026 Polymarket Prediction Engine — Build Plan

## Summary

A hybrid statistical + LLM engine that produces calibrated probabilities for World Cup 2026 markets (match winners, exact scores, tournament outright, top scorer, golden boot), compares them against live Polymarket prices, and pushes any positive-edge opportunities to a Telegram bot. Backend runs 24/7 on a free-tier cloud VM; consumed via a mobile-installable PWA dashboard.

**Constraint:** WC 2026 kicks off June 11, 2026 in Mexico. Build window is ~3 weeks from today (May 23).
**Total cost:** $0/month with the free-tier stack below.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DATA SOURCES                         │
│  Sofascore │ FBref │ football-data │ RSS │ Reddit │ PM  │
└────┬─────────┬──────────┬────────────┬──────┬───────┬───┘
     │         │          │            │      │       │
     ▼         ▼          ▼            ▼      ▼       ▼
┌─────────────────────────────────────────────────────────┐
│              INGESTION LAYER (Python)                   │
│  Scrapers · RSS pollers · Polymarket CLOB client        │
└─────────┬───────────────────────────────────────────────┘
          ▼
┌─────────────────────────────────────────────────────────┐
│              STORAGE (Supabase Postgres)                │
│  teams · players · matches · news · predictions · edges │
└─────────┬───────────────────────────────────────────────┘
          ▼
┌─────────────────────────────────────────────────────────┐
│                 MODELING LAYER                          │
│  1. Dixon-Coles match model (attack/defense ratings)    │
│  2. Squad strength adjuster (26-man depth weighted)     │
│  3. LLM context layer (Groq Llama — news → rating Δ)     │
│  4. Monte Carlo tournament sim (10k runs)               │
│  5. Player-level top scorer / golden boot model         │
└─────────┬───────────────────────────────────────────────┘
          ▼
┌─────────────────────────────────────────────────────────┐
│              EDGE CALCULATOR + ALERTING                 │
│  model_prob − PM_implied_prob = edge                    │
│  Any positive edge → Telegram push + dashboard          │
└────────────────┬───────────────────────┬────────────────┘
                 ▼                       ▼
        ┌──────────────────┐   ┌──────────────────────┐
        │  TELEGRAM BOT    │   │  NEXT.JS PWA         │
        │  (push alerts)   │   │  (dashboard on phone)│
        └──────────────────┘   └──────────────────────┘
```

---

## Stack

| Layer | Tool | Free tier |
|---|---|---|
| Compute | Oracle Cloud Free Tier ARM VM | 24GB RAM, forever free |
| Database | Supabase | 500MB Postgres, free |
| Primary LLM | Mistral AI (mistral-small-latest) | EU-based, free tier, no regional limits |
| Backup LLM | Google Gemini (optional, where available) | 1M tokens/day |
| Dashboard host | Vercel | Free for hobby |
| Alerts | Telegram Bot API | Free |
| Scheduling | APScheduler in-process + cron | — |
| Error monitoring | Sentry | Free tier (5k events/mo) |

---

## Data Layer

### Sources

**Sofascore** (scraping undocumented API)
- Per-match player ratings (0–10) + recent form, minutes, position
- Polite rate limit: 1 req/sec with jitter, rotating user-agents
- Cache aggressively (player ratings only change once per match)

**FBref** (scraping, static HTML — more reliable)
- Club-season underlying stats (xG, xA, shot quality, defensive actions, GK saves%)
- Used for player quality baseline + Sofascore fallback if blocked

**football-data.org** (free API tier)
- Fixtures, results, lineups when posted
- 10 req/min — plenty

**Polymarket CLOB API**
- All WC markets and order books, free + public
- Active match markets: poll 60s. Outrights / futures: poll 5 min.

**News**
- RSS: BBC Sport, Guardian Football, ESPN FC, The Athletic snippets
- Reddit: r/soccer + national team subs via free JSON endpoints (`.json` suffix)
- Poll every 5 min

### Storage schema (Postgres)

```sql
teams              (id, name, fifa_rank, base_attack, base_defense, ...)
players            (id, team_id, name, position, club, sofascore_id, rating_avg, ...)
squads             (team_id, player_id, called_up, starter_prob, status, fitness)
matches            (id, home_id, away_id, date, venue, stage, ...)
match_predictions  (match_id, model_version, p_home, p_draw, p_away,
                    score_distribution_json, computed_ts)
news_events        (id, source, headline, body, fetched_ts, relevance_score,
                    affected_team_ids, affected_player_ids, llm_summary)
rating_deltas      (id, source_news_id, team_id, attack_delta, defense_delta,
                    confidence, applied_ts, expires_ts)
polymarket_prices  (market_id, outcome, price, ts)
edges              (market_id, outcome, model_prob, pm_prob, edge_pct, ts, alerted)
```

---

## Modeling Layer

### 1. Base match model — Dixon-Coles

Bivariate Poisson with low-score correlation adjustment (the well-known fix for 0-0/1-0/0-1/1-1 happening more than independent Poisson predicts). Each team gets:
- `λ_attack`: scoring rate
- `μ_defense`: conceding rate
- shared `ρ` tau parameter for low-score correlation

Fitted on all international matches 2022–present with:
- Recency weights (exponential half-life ~18 months)
- Competition weights (WC = 1.0, continental = 0.85, qualifier = 0.7, friendly = 0.4)
- Neutral-venue handling (almost all WC matches except hosts')

**Outputs:** P(home), P(draw), P(away), full score distribution up to 5-5.

### 2. Squad strength adjuster

Replace static team ratings with dynamic ones derived from the actual 26-man squad:

```
team_attack_rating = Σ over squad:
    starter_prob[p] × position_weight[p] ×
    (0.7 × club_xG90[p] + 0.3 × intl_xG90[p])
```

Similar formula for defense using defensive actions + GK saves%. Position weights tuned in backtest; rough starting values — attackers 50%, mids 30%, defenders 15%, GK 5% (varies by metric type).

Rotation across the tournament: starter probabilities decay slightly for outfielders by minutes load, with managers known to rotate (Brazil, France) modeled more aggressively than those who don't (England under most recent managers).

### 3. LLM context layer (Groq Llama 3.3 70B)

Two stages.

**Stage A — Relevance classifier** (every news/Reddit item)
- Prompt: classify whether the item materially affects any WC team's prediction
- Output JSON: `{ affects: bool, teams: [...], players: [...], category: injury|lineup|tactical|morale|suspension, severity: 0-10, summary: "..." }`
- Items scoring `affects: false` are stored but skip Stage B

**Stage B — Rating adjuster** (items that pass Stage A)
- Inputs: news item + current squad + current base ratings
- Output: rating deltas with confidence + expiry timestamp + plain-English reasoning (logged for audit)
- **Guardrails:** per-event delta capped at ±5% of base rating. Big moves (>3%) require 2 independent sources within 24h.
- Deltas decay back to zero across their expiry window. Example: "minor knock, doubtful for Game 1" → delta expires after Game 1.

### 4. Tournament Monte Carlo simulator

For each of 10,000 runs:
1. Sample each group-stage match score from its Dixon-Coles distribution
2. Resolve group standings with full FIFA tiebreaker chain: points → goal diff → goals scored → H2H points → H2H GD → H2H GF → fair play → drawing of lots
3. Build knockout bracket per FIFA's seeding rules
4. Simulate knockouts: regulation → extra time (lambda ×0.33) → shootout (50/50, small GK-rating tilt)
5. Record: champion, finalists, top-4, each team's furthest stage

**Outputs:** P(team wins outright), P(team reaches each stage), P(specific final matchup), etc.

### 5. Top scorer / Golden Boot model

Player-level Poisson on goals/90, fitted on club + international goals (recency weighted, competition-strength adjusted).

For each MC run:
- Sample player's minutes per match (starter prob + expected rotation)
- Sample goals from rate × minutes/90
- Accumulate across simulated tournament path (player only scores in games their team plays)
- Top scorer = arg max across all players for that run

Adjustments: penalty taker (~+0.1 g/match), set-piece taker (small bump), known finishers with above-rate conversion (small bump from xG overperformance trend).

**Honest caveat:** top scorer has enormous variance. The model will be roughly calibrated on probabilities but don't expect any single player to be >15% likely. Treat as a market for spotting *mispriced* names rather than picking a winner.

---

## Edge Calculator (Polymarket)

For each WC market we track:
1. Map outcome to model prediction (manual mapping table, ~30–50 markets at most)
2. Pull `model_prob` from latest predictions table
3. Pull `pm_prob` from last-traded price or order-book midpoint
4. `edge = model_prob − pm_prob`
5. If edge > 0 AND lower bound of confidence band > 0 → flag

Confidence bands:
- Match markets: bootstrap on Dixon-Coles parameters
- Tournament markets: MC sim variance across 10k runs

---

## Alerting

**Telegram bot:**
- Format: `🔥 NEW EDGE: [Outcome] @ [PM price] — model [X%] — edge +Y% — [link]`
- Include top 2 news items driving current model state (transparency)
- Cooldown: don't re-alert same market within 30 min unless edge widens by ≥2%

**PWA dashboard (Next.js on Vercel):**
- Live edge table sorted by size
- Per-team page: current ratings, squad, news items affecting prediction, next-match probabilities
- Per-match page: score-distribution heatmap, model vs. market side-by-side
- News feed showing which items moved which ratings and by how much (audit trail)
- Install to home screen for native-app feel

---

## Validation Plan

Backtests run in this order. Don't go live until all phases pass.

**Phase 1 — Euros 2024 + Copa America 2024**
- Most relevant format, recent player data
- Walk-forward: for each match, reconstruct what model would have known pre-kickoff
- Metrics: Brier score on 1X2, log loss, calibration plot (predicted-prob bins vs. actual frequency)
- Benchmark: closing line from Pinnacle. Beating closing line = the gold standard.

**Phase 2 — World Cups 2014, 2018, 2022**
- Tournament-specific dynamics (group→knockout transitions, fatigue, surprise runs)
- Verify MC sim tournament-winner probs against eventual outcomes
- Top-scorer model checked against Müller (2014), Kane (2018), Mbappé (2022)

**Phase 3 — WC 2026 warmup friendlies + final qualifiers**
- Calibrate current-form weights
- Sanity check confederation strength (CONCACAF/AFC over- or under-priced?)

**Phase 4 — Live shadow mode**
- Run against first 3–5 WC 2026 matches without betting, verify calibration holds
- Then enable live alerts

**Go-live gate:** Brier ≤ 0.21 on Euros 2024, calibration plot within 5pp of diagonal in all probability bins. If not, debug before turning on alerts.

---

## 3-Week Build Timeline

### Week 1 (May 23 – May 30): Data + Base Model
- D1–2: Provision Oracle VM, Supabase, project skeleton, CI
- D3–4: Sofascore + FBref scrapers, schema, initial backfill
- D5: football-data.org integration, Polymarket CLOB client
- D6–7: Dixon-Coles base model fitted on 2022–2026 international data + Euros 2024 backtest pass 1

### Week 2 (May 31 – June 6): LLM Layer + Tournament Sim + Validation
- D8: RSS/Reddit pollers running
- D9: Groq Stage A classifier + prompt iteration
- D10–11: Groq Stage B rating adjuster, squad strength adjuster, ingest 26-man squads (FIFA deadline ~June 1)
- D12: Tournament MC sim + top scorer model
- D13–14: Full backtests (Euros 2024, Copa 2024, WCs 2014/18/22), calibration tuning

### Week 3 (June 7 – June 10): Frontend + Alerts + Hardening
- D15–16: Next.js PWA dashboard (edges, match pages, team pages, news audit)
- D17: Telegram bot + edge alerting with cooldowns
- D18: Final calibration on warmup friendlies + announced squads
- D19: Stress test, system monitoring (Sentry), deploy
- D20 (June 10): Dry run with locked-in lineups for opener; live from kickoff June 11

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Sofascore blocks scraper | FBref fallback, aggressive caching |
| Mistral rate limits during news bursts | Queue + batch; degrade to keyword classifier on overflow |
| Model poorly calibrated at go-live | Validation gates; shadow mode for 3–5 matches |
| LLM hallucinates news impact | Hard cap on per-event delta; 2-source rule for big moves; full reasoning log |
| Polymarket market too thin to act on | Show order-book depth in alert; include available size |
| MC sim bug (esp. tiebreakers) | Unit tests against historical groups with known final tables |
| Oracle VM reclaimed for inactivity | Daily snapshot to Supabase storage; keepalive cron; redeploy script |
| Squad announcements late / leaked unofficially | Pull confirmed list from FIFA + cross-check 2 outlets before applying |

---

## Out of Scope (v1)

- Auto-execution of bets (manual click-through only)
- In-play / live match predictions (pre-match only for v1)
- Markets outside the World Cup
- Native mobile app (PWA is the mobile experience)
- Multi-user / auth (single-user deployment)

---

## Open Questions Before Building

1. Where should the repo live (your GitHub)?
2. Do you have a Telegram account ready, or want a different alert channel?
3. Any teams you'll follow closely? Their models can be tuned/QA'd first.
4. OK to start scaffolding the repo now, or want to revise anything in this plan first?
