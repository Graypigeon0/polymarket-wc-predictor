# Portfolio Analyzer — Design

Date: 2026-10-30

## Purpose

A command-line tool where the user enters a portfolio (stocks, ETFs, commodities such as oil, and crypto such as bitcoin) and gets, per holding: a bullish outlook, a bearish outlook, a candlestick reading, and suggested stop losses. For the portfolio as a whole it judges whether the hedge is effective. It also suggests equities that screen as underpriced.

It is decision support, not financial advice. All numbers come from deterministic code; an LLM is optional and only narrates.

## Usage

```
python -m portfolio_analyzer AAPL:40 MSFT:30 XOM:15 CL=F:5 BTC-USD:10 \
    --benchmark SPY --report report.md [--narrate] [--universe sp500|FILE]
```

- `TICKER:WEIGHT` — weight is a percent of the portfolio. Omitted weights default to equal.
- Any Yahoo Finance symbol is accepted: equities/ETFs (`AAPL`, `USO`), futures (`CL=F` crude oil, `GC=F` gold), crypto (`BTC-USD`, `ETH-USD`, `ETC-USD`).
- Default benchmark `SPY`; default underpriced universe is the S&P 500, overridable with a file of tickers.

## Asset classes

Each ticker is classified (equity, etf, commodity, crypto) from the data provider's quote type. Behaviour by class:

| Component | Equity | ETF | Commodity | Crypto |
|---|---|---|---|---|
| Candlesticks | yes | yes | yes | yes |
| Stop loss | yes | yes | yes | yes |
| Hedge analysis | yes | yes | yes | yes |
| Valuation / underpriced | yes | no | no | no |

Non-equities are reported as "valuation n/a (no earnings/cash flow)" rather than guessed at. Crypto trades 365 days a year, so annualisation uses 365 for crypto and 252 for everything else.

## Components

New package `portfolio_analyzer/` at the repo root, independent of the World Cup code.

1. **data** — downloads and caches OHLCV history, fundamentals, sector and asset class via yfinance. Every other component takes DataFrames, so they are testable offline.
2. **candles** — detects doji, hammer, shooting star, bullish/bearish engulfing, morning/evening star, three white soldiers / three black crows. Outputs a bullish/bearish/neutral reading plus trend context (20/50-day MAs, RSI).
3. **stops** — two methods: ATR-based (2 x 14-day ATR below price) and just below the nearest swing-low support. Shows both with percentage risk; recommends the tighter one when it is still outside normal daily noise (more than 1 ATR away). Volatile assets such as crypto naturally get wider stops.
4. **hedge** — portfolio-level: beta vs benchmark, correlation matrix with flags on highly correlated pairs, sector/asset-class concentration, holdings that offset others (negative correlation, inverse or defensive assets), and historical stress tests (e.g. March 2020, 2022) comparing the portfolio to the benchmark where history exists. Verdict: effective / partial / none, with the reason.
5. **valuation** — screens the universe (equities only) on P/E and PEG against the sector median, free cash flow yield, and position in the 52-week range. Requires positive earnings and a non-broken trend to limit value traps. Returns the top candidates with reasons. These are candidates to research, not buy recommendations.
6. **outlook** — rule-based bullish and bearish bullets per holding, derived from the signals above. With `--narrate` and `ANTHROPIC_API_KEY` set, Claude writes a short paragraph from the same facts; it is given only computed figures and cannot add its own.
7. **report** — renders to the terminal and a Markdown file. Every report carries a "not financial advice; signals are heuristics" notice.

Data flow: tickers -> data -> (candles, stops, valuation, hedge) -> outlook -> report.

## Error handling

- Invalid ticker or missing data: reported for that holding; the rest of the portfolio still runs.
- Missing fundamentals: stock skipped in the valuation screen and flagged.
- Short history (e.g. a new coin): stress tests that predate the data are skipped and noted.
- Data provider unreachable: clear error naming the cause; no partial silent output.

## Testing

Each component is tested on small hand-built series with known answers (a constructed hammer, a known beta, a known ATR). One network smoke test checks that the live fetch works. The LLM narration is tested with the client stubbed.

## Risks and early checks

- The sandbox reaches the internet through a proxy; the first implementation task verifies Yahoo Finance is reachable. If not, add a fallback data source before building further.
- yfinance is an unofficial API and can break or rate-limit; data access is isolated in `data` so it can be swapped.

## Out of scope

Options-based hedges, intraday data, backtesting the signals, a web UI, order execution.
