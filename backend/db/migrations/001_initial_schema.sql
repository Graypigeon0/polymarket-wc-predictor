-- =====================================================================
-- polymarket-wc-predictor :: initial schema
-- =====================================================================
-- All FKs use ON DELETE CASCADE where the child rows are meaningless
-- without the parent. Timestamps are timestamptz; everything in UTC.
-- =====================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ---------- teams ----------------------------------------------------
CREATE TABLE teams (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    fifa_code       TEXT UNIQUE NOT NULL,           -- e.g. 'BRA'
    name            TEXT NOT NULL,
    confederation   TEXT NOT NULL,                  -- UEFA/CONMEBOL/CONCACAF/AFC/CAF/OFC
    fifa_rank       INT,
    base_attack     DOUBLE PRECISION,               -- Dixon-Coles fitted
    base_defense    DOUBLE PRECISION,
    home_adv        DOUBLE PRECISION DEFAULT 0.0,   -- relevant for host nations
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ---------- players --------------------------------------------------
CREATE TABLE players (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    team_id          UUID REFERENCES teams(id) ON DELETE CASCADE,
    name             TEXT NOT NULL,
    position         TEXT NOT NULL,                 -- GK/DEF/MID/ATT
    club             TEXT,
    date_of_birth    DATE,
    sofascore_id     BIGINT UNIQUE,
    fbref_id         TEXT UNIQUE,
    rating_avg       DOUBLE PRECISION,              -- recent Sofascore average
    club_xg90        DOUBLE PRECISION,              -- attackers/mids
    club_xga90       DOUBLE PRECISION,              -- defenders/GK
    intl_g90         DOUBLE PRECISION,              -- international scoring rate
    intl_xg90        DOUBLE PRECISION,
    penalty_taker    BOOLEAN DEFAULT FALSE,
    set_piece_taker  BOOLEAN DEFAULT FALSE,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_players_team ON players(team_id);

-- ---------- squads (26-man for WC) -----------------------------------
CREATE TABLE squads (
    team_id          UUID REFERENCES teams(id) ON DELETE CASCADE,
    player_id        UUID REFERENCES players(id) ON DELETE CASCADE,
    called_up        BOOLEAN DEFAULT TRUE,
    starter_prob     DOUBLE PRECISION DEFAULT 0.5,  -- 0..1
    status           TEXT DEFAULT 'available',      -- available/doubtful/out/suspended
    fitness          DOUBLE PRECISION DEFAULT 1.0,  -- 0..1, decays with minutes load
    notes            TEXT,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (team_id, player_id)
);

-- ---------- matches --------------------------------------------------
CREATE TABLE matches (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    home_id         UUID NOT NULL REFERENCES teams(id),
    away_id         UUID NOT NULL REFERENCES teams(id),
    kickoff         TIMESTAMPTZ NOT NULL,
    venue           TEXT,
    stage           TEXT NOT NULL,                  -- group/r16/qf/sf/3p/final/friendly/qual
    competition     TEXT NOT NULL,                  -- 'WC2026', 'EURO2024', etc.
    is_neutral      BOOLEAN DEFAULT TRUE,
    home_goals      INT,                            -- NULL until played
    away_goals      INT,
    completed       BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_matches_kickoff ON matches(kickoff);
CREATE INDEX idx_matches_competition ON matches(competition);

-- ---------- match predictions ---------------------------------------
CREATE TABLE match_predictions (
    id                     UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    match_id               UUID NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    model_version          TEXT NOT NULL,
    p_home                 DOUBLE PRECISION NOT NULL,
    p_draw                 DOUBLE PRECISION NOT NULL,
    p_away                 DOUBLE PRECISION NOT NULL,
    expected_home_goals    DOUBLE PRECISION,
    expected_away_goals    DOUBLE PRECISION,
    score_distribution     JSONB,                    -- {"1-0": 0.12, "2-1": 0.09, ...}
    confidence_band        JSONB,                    -- bootstrap CIs
    computed_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_predictions_match ON match_predictions(match_id, computed_at DESC);

-- ---------- tournament predictions (MC sim aggregates) --------------
CREATE TABLE tournament_predictions (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    team_id             UUID NOT NULL REFERENCES teams(id),
    model_version       TEXT NOT NULL,
    p_win_outright      DOUBLE PRECISION,
    p_reach_final       DOUBLE PRECISION,
    p_reach_semi        DOUBLE PRECISION,
    p_reach_qf          DOUBLE PRECISION,
    p_reach_r16         DOUBLE PRECISION,
    p_win_group         DOUBLE PRECISION,
    p_advance_group     DOUBLE PRECISION,
    n_simulations       INT NOT NULL,
    computed_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_tournament_team ON tournament_predictions(team_id, computed_at DESC);

-- ---------- top scorer predictions ----------------------------------
CREATE TABLE top_scorer_predictions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    player_id       UUID NOT NULL REFERENCES players(id),
    model_version   TEXT NOT NULL,
    p_top_scorer    DOUBLE PRECISION,
    expected_goals  DOUBLE PRECISION,
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_top_scorer_player ON top_scorer_predictions(player_id, computed_at DESC);

-- ---------- news events ---------------------------------------------
CREATE TABLE news_events (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source              TEXT NOT NULL,             -- 'bbc'|'guardian'|'reddit'|...
    source_url          TEXT UNIQUE,
    headline            TEXT NOT NULL,
    body                TEXT,
    fetched_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    classified          BOOLEAN DEFAULT FALSE,
    affects_model       BOOLEAN,
    relevance_score     INT,                       -- 0..10 from Stage A LLM
    category            TEXT,                      -- injury|lineup|tactical|morale|suspension
    affected_team_ids   UUID[],
    affected_player_ids UUID[],
    llm_summary         TEXT
);
CREATE INDEX idx_news_fetched ON news_events(fetched_at DESC);
CREATE INDEX idx_news_classified ON news_events(classified) WHERE classified = FALSE;

-- ---------- rating deltas (LLM Stage B output) ----------------------
CREATE TABLE rating_deltas (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_news_id  UUID REFERENCES news_events(id) ON DELETE CASCADE,
    team_id         UUID REFERENCES teams(id),
    player_id       UUID REFERENCES players(id),
    attack_delta    DOUBLE PRECISION DEFAULT 0.0,
    defense_delta   DOUBLE PRECISION DEFAULT 0.0,
    confidence      DOUBLE PRECISION,              -- 0..1
    reasoning       TEXT,                          -- LLM's natural-language audit trail
    applied_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at      TIMESTAMPTZ NOT NULL,
    superseded      BOOLEAN DEFAULT FALSE
);
CREATE INDEX idx_deltas_active ON rating_deltas(team_id, expires_at)
    WHERE superseded = FALSE;

-- ---------- polymarket prices ---------------------------------------
CREATE TABLE polymarket_markets (
    id              TEXT PRIMARY KEY,              -- polymarket condition_id or token_id
    market_type     TEXT NOT NULL,                 -- 'match_1x2'|'exact_score'|'outright'|'top_scorer'|...
    description     TEXT,
    match_id        UUID REFERENCES matches(id),
    team_id         UUID REFERENCES teams(id),
    player_id       UUID REFERENCES players(id),
    outcome_label   TEXT,                          -- e.g. 'BRA wins', '2-1', 'Mbappe top scorer'
    active          BOOLEAN DEFAULT TRUE,
    discovered_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE polymarket_prices (
    id            BIGSERIAL PRIMARY KEY,
    market_id     TEXT NOT NULL REFERENCES polymarket_markets(id) ON DELETE CASCADE,
    price         DOUBLE PRECISION NOT NULL,       -- last-traded or midpoint
    bid           DOUBLE PRECISION,
    ask           DOUBLE PRECISION,
    book_depth    DOUBLE PRECISION,                -- USD available within 1pp of midpoint
    captured_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_pm_prices_recent ON polymarket_prices(market_id, captured_at DESC);

-- ---------- edges ---------------------------------------------------
CREATE TABLE edges (
    id              BIGSERIAL PRIMARY KEY,
    market_id       TEXT NOT NULL REFERENCES polymarket_markets(id),
    model_prob      DOUBLE PRECISION NOT NULL,
    pm_prob         DOUBLE PRECISION NOT NULL,
    edge            DOUBLE PRECISION NOT NULL,     -- model_prob - pm_prob
    edge_lower_ci   DOUBLE PRECISION,              -- lower bound of confidence band
    model_version   TEXT NOT NULL,
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    alerted         BOOLEAN DEFAULT FALSE,
    alerted_at      TIMESTAMPTZ
);
CREATE INDEX idx_edges_recent ON edges(computed_at DESC);
CREATE INDEX idx_edges_positive ON edges(edge DESC) WHERE edge > 0;
