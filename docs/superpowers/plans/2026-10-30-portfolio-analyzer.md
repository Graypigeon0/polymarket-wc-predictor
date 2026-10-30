# Portfolio Analyzer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A CLI that takes a portfolio of stocks/ETFs/commodities/crypto and reports per-holding bull/bear outlook, candlestick reading and stop losses, a portfolio hedge verdict, and underpriced-stock candidates.

**Architecture:** New package `portfolio_analyzer/` beside `backend/`, sharing nothing with it. `data` is the only module that touches the network; every analysis module is a pure function over pandas DataFrames, so all of it is tested offline on hand-built series. `outlook` turns computed signals into bullets by rule; Claude narration is optional and sees only computed figures.

**Tech Stack:** Python >=3.11, pandas, numpy, yfinance (new dependency), typer + rich (already in `pyproject.toml`), optional `anthropic` extra, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-30-portfolio-analyzer-design.md`

## Global Constraints

- Python >=3.11; ruff line length 100 (`pyproject.toml`); run `ruff check portfolio_analyzer` before each commit.
- Default benchmark `SPY`; default underpriced universe is the S&P 500, overridable with a file of tickers (one per line).
- Annualisation factor: 365 for `crypto`, 252 for every other asset class.
- ATR stop = price − 2 × 14-day ATR. Support stop = just below the nearest swing-low. Recommend the tighter of the two only if it is more than 1 ATR below price; otherwise recommend the ATR stop.
- Candlestick patterns: doji, hammer, shooting star, bullish engulfing, bearish engulfing, morning star, evening star, three white soldiers, three black crows. Trend context from 20/50-day SMAs and 14-day RSI.
- Valuation applies to `equity` only; other classes report `valuation n/a (no earnings/cash flow)`.
- Stress windows: `covid_2020` = 2020-02-19..2020-03-23, `bear_2022` = 2022-01-03..2022-10-12. Skip (and note) any window the holdings' history does not cover.
- Every report carries: `Not financial advice. Signals are heuristics, not predictions.`
- One bad ticker never aborts the run; it is reported for that holding only.
- No network access outside `data.py` and `valuation.load_universe`.

## Review Focus

1. A new asset with under 60 bars of history (recent coin): reports `insufficient data` for candles/stops instead of crashing — Task 3, Task 4.
2. Weights that do not sum to 100 (`AAPL:40 MSFT:30`): normalised to 1.0 with a warning; mixing weighted and unweighted tickers is an error — Task 1.
3. Duplicate or lowercase tickers (`aapl AAPL`): uppercased; duplicates rejected with a clear error — Task 1.
4. Crypto trades weekends, equities do not: returns are inner-joined on dates before beta/correlation, and annualisation uses 365 only for crypto — Task 5.
5. Single-holding portfolio and flat (zero-volatility) price series: no division-by-zero, RSI/beta defined, hedge verdict `none` — Task 3, Task 5.

---

### Task 1: Scaffold, models and holdings parsing

**Files:**
- Create: `portfolio_analyzer/__init__.py`, `portfolio_analyzer/models.py`, `portfolio_analyzer/holdings.py`, `portfolio_analyzer/tests/__init__.py`, `portfolio_analyzer/tests/test_holdings.py`
- Modify: `pyproject.toml` (add `yfinance>=0.2.40` to dependencies; `[project.optional-dependencies] narrate = ["anthropic>=0.40"]`; add `"portfolio_analyzer"` to `[tool.hatch.build.targets.wheel] packages`; `testpaths = ["backend/tests", "portfolio_analyzer/tests"]`; register pytest marker `network`; script `portfolio-analyzer = "portfolio_analyzer.cli:app"`), `.gitignore` (add `.cache/`)

**Interfaces:**
- Produces (`models.py`, all `@dataclass`):
  - `AssetClass = Literal["equity", "etf", "commodity", "crypto"]`
  - `Holding(ticker: str, weight: float)`
  - `TickerData(ticker: str, asset_class: AssetClass, prices: pd.DataFrame, fundamentals: dict[str, Any])` — `prices` has columns `Open, High, Low, Close, Volume` and a `DatetimeIndex`
  - `PatternHit(name: str, direction: Literal["bullish","bearish","neutral"], date: pd.Timestamp)`
  - `CandleReading(ticker: str, patterns: list[PatternHit], bias: Literal["bullish","bearish","neutral"], trend: Literal["up","down","sideways"], rsi: float | None, note: str | None)`
  - `StopSuggestion(ticker: str, atr_stop: float | None, support_stop: float | None, recommended: float | None, risk_pct: float | None, method: Literal["atr","support","n/a"], note: str | None)`
  - `StressResult(name: str, portfolio_return: float | None, benchmark_return: float | None, skipped_reason: str | None)`
  - `HedgeResult(beta: float, correlation: pd.DataFrame, high_corr_pairs: list[tuple[str,str,float]], concentration: dict[str, float], offsets: list[str], stress: list[StressResult], verdict: Literal["effective","partial","none"], reason: str)`
  - `ValuationCandidate(ticker: str, score: float, reasons: list[str])`
  - `Outlook(ticker: str, bullish: list[str], bearish: list[str], narrative: str | None)`
- Produces (`holdings.py`): `parse_holdings(args: list[str]) -> tuple[list[Holding], list[str]]` — returns holdings (weights sum to 1.0) and warnings.

- [ ] **Step 1: Create the feature branch** — `git checkout -b feature/portfolio-analyzer` (repo is on a detached HEAD).
- [ ] **Step 2: Write failing tests in `test_holdings.py`**: `test_weights_normalised` (`["AAPL:40","MSFT:30"]` → weights 4/7 and 3/7, one warning); `test_equal_weights_when_omitted` (`["AAPL","BTC-USD"]` → 0.5 each, no warning); `test_mixed_weighted_unweighted_raises` (`ValueError` mentioning "all or none"); `test_uppercases` (`"aapl:50","cl=f:50"` → `AAPL`, `CL=F`); `test_duplicate_raises` (`ValueError` naming the ticker); `test_futures_and_crypto_symbols_parse` (`"CL=F:5"`, `"BTC-USD:10"`); `test_bad_weight_raises` (`"AAPL:abc"`, `"AAPL:-5"`); `test_empty_raises`.
- [ ] **Step 3: Run** `pytest portfolio_analyzer/tests/test_holdings.py -v` — expect FAIL (import error).
- [ ] **Step 4: Implement** `models.py` and `parse_holdings`. Split on the **last** colon so `CL=F:5` works; weights are percentages.
- [ ] **Step 5: Run** the same command — expect all PASS; `pytest backend/tests -q` still passes.
- [ ] **Step 6: Commit** `feat(portfolio): scaffold package, models and holdings parsing`

---

### Task 2: Data layer and connectivity check

**Files:**
- Create: `portfolio_analyzer/data.py`, `portfolio_analyzer/tests/test_data.py`

**Interfaces:**
- Consumes: `TickerData`, `AssetClass` from `models.py`.
- Produces:
  - `classify(quote_type: str, ticker: str) -> AssetClass` — `CRYPTOCURRENCY`→`crypto`, `FUTURE`→`commodity`, `ETF`→`etf`, `EQUITY`→`equity`; unknown types with a `=F` suffix → `commodity`, `-USD` suffix → `crypto`, else `equity`.
  - `fetch_ticker(ticker: str, start: str = "2019-12-01", cache_dir: Path = Path(".cache/portfolio_analyzer")) -> TickerData` — raises `DataError` on empty/invalid data; caches per ticker per day (parquet or pickle in `cache_dir`).
  - `class DataError(Exception)`
  - `annualisation_factor(asset_class: AssetClass) -> int` — 365 for crypto else 252.

- [ ] **Step 1: Connectivity check first.** Run `python -c "import yfinance as yf; print(yf.Ticker('SPY').history(period='5d'))"`. If it fails (proxy/403/empty), **stop and report to the user** with the error and propose a fallback source; do not continue to Task 3.
- [ ] **Step 2: Write failing tests** with `yfinance.Ticker` monkeypatched: `test_classify_quote_types`, `test_classify_suffix_fallbacks`, `test_annualisation_factor` (crypto 365, equity/etf/commodity 252), `test_fetch_drops_nan_rows_and_sorts_index`, `test_fetch_empty_raises_dataerror`, `test_cache_hit_skips_second_download` (second call makes no patched call). Add one `@pytest.mark.network` smoke test `test_live_fetch_spy` asserting ≥200 rows and the five columns.
- [ ] **Step 3: Run** `pytest portfolio_analyzer/tests/test_data.py -v -m "not network"` — expect FAIL.
- [ ] **Step 4: Implement** `data.py`. Pull fundamentals from `Ticker.info` keys `trailingPE, pegRatio, freeCashflow, marketCap, sector, fiftyTwoWeekHigh, fiftyTwoWeekLow`, missing keys stay absent.
- [ ] **Step 5: Run** the non-network tests (PASS), then `pytest -m network portfolio_analyzer/tests/test_data.py` (PASS).
- [ ] **Step 6: Commit** `feat(portfolio): yfinance data layer with cache`

---

### Task 3: Candlestick analysis

**Files:**
- Create: `portfolio_analyzer/candles.py`, `portfolio_analyzer/tests/test_candles.py`

**Interfaces:**
- Consumes: `TickerData`, `CandleReading`, `PatternHit`.
- Produces: `analyse_candles(data: TickerData) -> CandleReading`; `rsi(close: pd.Series, period: int = 14) -> float | None`.

Pattern definitions to pin: doji = body ≤ 10% of range; hammer = lower shadow ≥ 2×body, upper shadow ≤ body, occurring after a down-trend (20-SMA falling); shooting star = mirror, after an up-trend; engulfing = body fully contains previous opposite-colour body; stars = three-bar reversals with the middle body ≤ 30% of the first bar's body; soldiers/crows = three consecutive same-colour bars each closing beyond the previous close. `bias` = side with more hits among the last 3 bars (tie → `neutral`). `trend`: `up` if close > SMA20 > SMA50, `down` if close < SMA20 < SMA50, else `sideways`.

- [ ] **Step 1: Write failing tests** on hand-built OHLC frames: `test_detects_hammer_after_downtrend`, `test_detects_bullish_engulfing`, `test_detects_evening_star`, `test_detects_three_white_soldiers`, `test_doji`, `test_no_patterns_is_neutral`, `test_trend_up_down_sideways`, `test_rsi_all_gains_is_100`, `test_flat_prices_rsi_is_none_not_crash` (zero volatility), `test_under_60_bars_returns_insufficient_data_note` (`bias=="neutral"`, `note` contains "insufficient data").
- [ ] **Step 2: Run** `pytest portfolio_analyzer/tests/test_candles.py -v` — expect FAIL.
- [ ] **Step 3: Implement** with vectorised pandas; guard zero range bars (`high == low`) and zero loss in RSI.
- [ ] **Step 4: Run** — expect PASS. Run `ruff check portfolio_analyzer`.
- [ ] **Step 5: Commit** `feat(portfolio): candlestick pattern detection`

---

### Task 4: Stop losses

**Files:**
- Create: `portfolio_analyzer/stops.py`, `portfolio_analyzer/tests/test_stops.py`

**Interfaces:**
- Consumes: `TickerData`, `StopSuggestion`.
- Produces: `atr(prices: pd.DataFrame, period: int = 14) -> pd.Series`; `suggest_stop(data: TickerData) -> StopSuggestion`. `risk_pct` = (price − recommended) / price × 100.

Swing-low = a bar whose Low is the lowest of the surrounding 5 bars on each side, within the last 60 bars, and below current price; use the nearest such low, stop = low × 0.99.

- [ ] **Step 1: Write failing tests**: `test_atr_known_values` (constant range 2 → ATR 2); `test_atr_stop_is_price_minus_two_atr`; `test_support_stop_below_swing_low` (constructed V-shaped series, stop = low×0.99); `test_recommends_tighter_when_beyond_one_atr`; `test_falls_back_to_atr_when_support_within_one_atr`; `test_no_swing_low_uses_atr_only`; `test_under_60_bars_returns_na_with_note`; `test_crypto_wider_stop_than_stable_series` (high-volatility series gets larger risk_pct).
- [ ] **Step 2: Run** `pytest portfolio_analyzer/tests/test_stops.py -v` — expect FAIL.
- [ ] **Step 3: Implement** both methods per the Global Constraints rule.
- [ ] **Step 4: Run** — expect PASS.
- [ ] **Step 5: Commit** `feat(portfolio): ATR and support stop losses`

---

### Task 5: Hedge analysis

**Files:**
- Create: `portfolio_analyzer/hedge.py`, `portfolio_analyzer/tests/test_hedge.py`

**Interfaces:**
- Consumes: `Holding`, `TickerData`, `HedgeResult`, `StressResult`, `annualisation_factor`.
- Produces: `analyse_hedge(holdings: list[Holding], data: dict[str, TickerData], benchmark: TickerData) -> HedgeResult`.

Rules to pin: daily returns from `Close`, **inner-joined on dates** across holdings and benchmark; portfolio return = weighted sum of holding returns; `beta` = cov(portfolio, benchmark)/var(benchmark); `high_corr_pairs` = pairs with ρ ≥ 0.85; `concentration` = weight by sector (equities) or asset class (others), flag any bucket >50% in `reason`; `offsets` = holdings with ρ ≤ −0.2 to the rest of the portfolio, or beta to benchmark < 0.3; stress return = compounded portfolio return over the window, skipped with `skipped_reason` if the joined data does not cover it. Verdict: **effective** if beta ≤ 0.7 and the portfolio beats the benchmark in every available stress window; **partial** if beta < 1.0 or it beats the benchmark in at least one window; else **none**. `reason` is a one-sentence explanation citing the figures.

- [ ] **Step 1: Write failing tests**: `test_beta_of_benchmark_clone_is_one`; `test_beta_of_half_cash_like_asset_is_about_half`; `test_negatively_correlated_holding_listed_as_offset`; `test_high_corr_pair_flagged`; `test_concentration_flags_over_50_percent`; `test_stress_window_skipped_when_history_short` (data starting 2023 → both windows skipped with reason); `test_stress_return_computed_when_covered`; `test_crypto_weekend_rows_dropped_by_inner_join` (BTC with Saturday/Sunday rows + equity weekdays → joined length equals weekday count, no NaNs); `test_single_holding_benchmark_clone_verdict_none`; `test_verdict_effective`, `test_verdict_partial`.
- [ ] **Step 2: Run** `pytest portfolio_analyzer/tests/test_hedge.py -v` — expect FAIL.
- [ ] **Step 3: Implement** per the pinned rules.
- [ ] **Step 4: Run** — expect PASS.
- [ ] **Step 5: Commit** `feat(portfolio): portfolio hedge analysis`

---

### Task 6: Underpriced screen

**Files:**
- Create: `portfolio_analyzer/valuation.py`, `portfolio_analyzer/tests/test_valuation.py`

**Interfaces:**
- Consumes: `TickerData`, `ValuationCandidate`, `fetch_ticker`.
- Produces: `load_universe(spec: str) -> list[str]` (`"sp500"` → scrape the Wikipedia S&P 500 table with `pandas.read_html`, replacing `.` with `-` in symbols; otherwise a path to a file of tickers, one per line, blank lines and `#` comments ignored); `screen_underpriced(universe: dict[str, TickerData], top: int = 10) -> list[ValuationCandidate]`.

Scoring to pin: only `equity` with positive `trailingPE` and positive `freeCashflow`, and `Close >= 0.9 × SMA200`. Within each sector, percentile-rank (0–1, higher = cheaper) low P/E, low PEG (when present), high FCF yield (`freeCashflow/marketCap`), and low position in the 52-week range; `score` = mean of the available ranks. `reasons` are human-readable strings such as `P/E 11.2 vs sector median 19.4`. Tickers failing fundamentals are returned in a separate `skipped` note by the caller, never guessed.

- [ ] **Step 1: Write failing tests**: `test_non_equity_excluded`, `test_negative_earnings_excluded`, `test_broken_trend_excluded` (close < 0.9×SMA200), `test_cheaper_stock_ranks_higher_within_sector`, `test_missing_peg_still_scored`, `test_missing_fundamentals_skipped_not_crashing`, `test_top_limits_results`, `test_load_universe_from_file` (comments/blank lines ignored), `test_load_universe_sp500_replaces_dots` (with `pandas.read_html` monkeypatched).
- [ ] **Step 2: Run** `pytest portfolio_analyzer/tests/test_valuation.py -v` — expect FAIL.
- [ ] **Step 3: Implement.** Fetching 500 tickers is slow: the caller fetches with a thread pool (8 workers); this module stays pure.
- [ ] **Step 4: Run** — expect PASS.
- [ ] **Step 5: Commit** `feat(portfolio): underpriced equity screen`

---

### Task 7: Outlook (rules + optional narration)

**Files:**
- Create: `portfolio_analyzer/outlook.py`, `portfolio_analyzer/tests/test_outlook.py`

**Interfaces:**
- Consumes: `TickerData`, `CandleReading`, `StopSuggestion`, `Outlook`.
- Produces: `build_outlook(data: TickerData, candles: CandleReading, stop: StopSuggestion) -> Outlook`; `narrate(outlook: Outlook, facts: dict[str, Any], client: Any, model: str = "claude-sonnet-5-5") -> str`.

Rules to pin (each emits a bullet naming its figure): bullish — trend `up`; RSI < 30 (oversold); bullish candle bias; price within 5% of 52-week low (equities/ETFs); equity P/E below sector median when available. Bearish — trend `down`; RSI > 70; bearish candle bias; price within 5% of 52-week high; stop risk_pct > 10 (high volatility); P/E above sector median. If a side has no signals, emit a single neutral line (`No strong bullish signals`). `narrate` sends only `facts` (the computed numbers and bullets) to `client.messages.create` and returns the text; the system prompt instructs it to use no figures other than those supplied.

- [ ] **Step 1: Write failing tests**: `test_oversold_adds_bullish_bullet_with_rsi_value`, `test_overbought_adds_bearish_bullet`, `test_trend_up_bullish_down_bearish`, `test_candle_bias_maps_to_side`, `test_no_signals_yields_neutral_line_each_side`, `test_commodity_skips_pe_bullets`, `test_narrate_passes_only_supplied_facts` (stub client records the prompt; assert it contains the RSI figure and no ticker not in facts), `test_narrate_returns_text`.
- [ ] **Step 2: Run** `pytest portfolio_analyzer/tests/test_outlook.py -v` — expect FAIL.
- [ ] **Step 3: Implement.** Import `anthropic` lazily inside the CLI only when `--narrate` is used.
- [ ] **Step 4: Run** — expect PASS.
- [ ] **Step 5: Commit** `feat(portfolio): rule-based outlook and optional narration`

---

### Task 8: Report, CLI and end-to-end

**Files:**
- Create: `portfolio_analyzer/report.py`, `portfolio_analyzer/cli.py`, `portfolio_analyzer/__main__.py`, `portfolio_analyzer/tests/test_report.py`, `portfolio_analyzer/tests/test_cli.py`
- Modify: `README.md` (add a "Portfolio analyzer" section with usage and the disclaimer)

**Interfaces:**
- Consumes: everything above.
- Produces: `render_markdown(...) -> str` in `report.py` taking holdings, per-ticker `Outlook`/`CandleReading`/`StopSuggestion`, `HedgeResult | None`, `list[ValuationCandidate]`, `errors: dict[str, str]`, `skipped_valuation: list[str]`, `warnings: list[str]`; `cli.app` (typer) with command signature `main(tickers: list[str], benchmark: str = "SPY", report: Path | None = None, narrate: bool = False, universe: str = "sp500")`. The data fetcher is injected via a module-level `fetch = data.fetch_ticker` so tests can replace it.

- [ ] **Step 1: Write failing tests**: `test_report_contains_disclaimer`; `test_report_has_section_per_holding_with_bull_bear_candles_stop`; `test_report_lists_failed_ticker_in_errors_and_still_renders_others`; `test_non_equity_shows_valuation_na`; `test_cli_end_to_end_with_fake_fetch` (CliRunner, synthetic data for 3 tickers incl. one `BTC-USD`; exit code 0, `--report` file written, hedge verdict line present); `test_cli_bad_ticker_does_not_abort` (fake fetch raises `DataError` for one); `test_cli_narrate_without_api_key_exits_with_clear_message`.
- [ ] **Step 2: Run** `pytest portfolio_analyzer/tests -v -m "not network"` — expect the new tests FAIL.
- [ ] **Step 3: Implement** report (rich for terminal, same content to Markdown) and CLI wiring; the underpriced section runs `screen_underpriced` over the universe fetched with a thread pool and excludes tickers already held.
- [ ] **Step 4: Run** the full suite: `pytest portfolio_analyzer/tests backend/tests -q -m "not network"` — expect all PASS; `ruff check portfolio_analyzer` clean.
- [ ] **Step 5: Live smoke run**: `python -m portfolio_analyzer AAPL:40 MSFT:30 XOM:15 CL=F:5 BTC-USD:10 --universe <small file of 10 tickers> --report /tmp/report.md`; confirm all sections render for the five holdings and that oil and bitcoin show `valuation n/a`.
- [ ] **Step 6: Commit** `feat(portfolio): report rendering and CLI`

---

## Self-review notes

- **Spec coverage:** usage/weights/benchmark/universe → Tasks 1, 8; asset-class table and 365/252 → Tasks 2, 5; candles → 3; stops → 4; hedge incl. stress tests → 5; valuation → 6; outlook and `--narrate` → 7; report + disclaimer → 8; error handling (bad ticker, missing fundamentals, short history, provider down) → Tasks 2, 3, 4, 5, 6, 8; connectivity risk → Task 2 Step 1; out-of-scope items intentionally absent.
- **Type consistency:** all dataclasses are defined once in Task 1 and referenced by those names in later tasks.
