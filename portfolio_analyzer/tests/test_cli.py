from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from portfolio_analyzer import cli
from portfolio_analyzer.data import DataError, DataProviderError
from portfolio_analyzer.models import TickerData
from portfolio_analyzer.outlook import NarrationError

runner = CliRunner()


def _td(ticker: str, asset_class: str = "equity", fundamentals=None, seed=None) -> TickerData:
    rng = np.random.default_rng(seed if seed is not None else len(ticker))
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 500))
    idx = pd.bdate_range("2019-12-02", periods=500)
    df = pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 1.0},
        index=idx,
    )
    return TickerData(ticker, asset_class, df, fundamentals or {})


def _fake_fetch(fail=None, provider_fail=()):
    fail = fail or {}

    def fake(ticker, **kw):
        if ticker in provider_fail:
            raise DataProviderError(f"cannot reach Yahoo for {ticker}")
        if ticker in fail:
            raise DataError(fail[ticker])
        kind = "crypto" if ticker.endswith("-USD") else "equity"
        return _td(ticker, kind)

    return fake


@pytest.fixture
def universe_file(tmp_path):
    p = tmp_path / "universe.txt"
    p.write_text("U1\nU2\nU3\n")
    return str(p)


def test_cli_end_to_end_with_fake_fetch(monkeypatch, tmp_path, universe_file):
    monkeypatch.setattr(cli, "fetch", _fake_fetch())
    out = tmp_path / "report.md"
    res = runner.invoke(
        cli.app, ["AAA:50", "BBB:30", "BTC-USD:20", "--universe", universe_file,
                  "--report", str(out)]
    )
    assert res.exit_code == 0, res.output
    md = out.read_text()
    assert "Hedge verdict:" in md
    assert "valuation n/a (no earnings/cash flow)" in md  # BTC-USD
    assert "Not financial advice. Signals are heuristics, not predictions." in md
    assert "No candidates found" in md
    assert "Hedge verdict" in res.output


def test_cli_bad_ticker_does_not_abort(monkeypatch, tmp_path, universe_file):
    monkeypatch.setattr(cli, "fetch", _fake_fetch(fail={"BAD": "No price data returned"}))
    out = tmp_path / "r.md"
    res = runner.invoke(
        cli.app, ["AAA:50", "BAD:50", "--universe", universe_file, "--report", str(out)]
    )
    assert res.exit_code == 0, res.output
    md = out.read_text()
    assert "- BAD: No price data returned" in md
    assert "## AAA (" in md
    assert "excluded from analysis (no data): BAD" in md  # hedge names the dropped ticker


def test_all_provider_failures_exit_nonzero(monkeypatch, universe_file):
    monkeypatch.setattr(cli, "fetch", _fake_fetch(provider_fail={"AAA", "BBB", "SPY"}))
    res = runner.invoke(cli.app, ["AAA", "BBB", "--universe", universe_file])
    assert res.exit_code != 0
    assert "could not be reached" in res.output
    assert "Portfolio analysis" not in res.output


def test_unexpected_exception_is_internal_error_not_provider_outage(monkeypatch, tmp_path):
    base = _fake_fetch()

    def fake(ticker, **kw):
        if ticker == "AAA":
            raise RuntimeError("yfinance API changed: unexpected keyword")
        return base(ticker, **kw)

    monkeypatch.setattr(cli, "fetch", fake)
    uni = tmp_path / "u.txt"
    uni.write_text("U1\n")
    out = tmp_path / "r.md"
    res = runner.invoke(cli.app, ["AAA", "BBB", "--universe", str(uni), "--report", str(out)])
    assert res.exit_code == 0, res.output
    md = out.read_text()
    assert "- AAA: internal error: RuntimeError: yfinance API changed" in md
    assert "Could not reach" not in md


def test_all_internal_errors_do_not_claim_provider_outage(monkeypatch, universe_file):
    def fake(ticker, **kw):
        raise RuntimeError("yfinance API changed: boom")

    monkeypatch.setattr(cli, "fetch", fake)
    res = runner.invoke(cli.app, ["AAA", "BBB", "--universe", universe_file])
    assert res.exit_code == 1
    assert "could not be reached" not in res.output
    assert "internal error" in res.output


def test_mixed_provider_failure_continues(monkeypatch, universe_file):
    monkeypatch.setattr(cli, "fetch", _fake_fetch(provider_fail={"BBB"}))
    res = runner.invoke(cli.app, ["AAA", "BBB", "--universe", universe_file])
    assert res.exit_code == 0, res.output
    assert "cannot reach Yahoo for BBB" in res.output


def test_benchmark_failure_makes_hedge_unavailable(monkeypatch, tmp_path, universe_file):
    monkeypatch.setattr(cli, "fetch", _fake_fetch(fail={"SPY": "no SPY"}))
    out = tmp_path / "r.md"
    res = runner.invoke(cli.app, ["AAA", "--universe", universe_file, "--report", str(out)])
    assert res.exit_code == 0, res.output
    md = out.read_text()
    assert "Hedge analysis unavailable: benchmark SPY unavailable" in md
    assert "## AAA (" in md


def test_bad_arguments_exit_2(universe_file):
    res = runner.invoke(cli.app, ["AAA:abc", "--universe", universe_file])
    assert res.exit_code == 2


def test_weight_warning_printed_and_in_report(monkeypatch, tmp_path, universe_file):
    monkeypatch.setattr(cli, "fetch", _fake_fetch())
    out = tmp_path / "r.md"
    res = runner.invoke(
        cli.app, ["AAA:30", "BBB:30", "--universe", universe_file, "--report", str(out)]
    )
    assert res.exit_code == 0, res.output
    assert "normalised to 100%" in res.output
    assert "normalised to 100%" in out.read_text()


def test_held_tickers_excluded_from_candidates(monkeypatch, tmp_path):
    f = {"trailingPE": 8.0, "freeCashflow": 1e9, "marketCap": 1e10, "sector": "Tech"}

    def fake(ticker, **kw):
        return _td(ticker, "equity", dict(f))

    monkeypatch.setattr(cli, "fetch", fake)
    uni = tmp_path / "u.txt"
    uni.write_text("AAA\nU1\nU2\n")
    out = tmp_path / "r.md"
    res = runner.invoke(cli.app, ["AAA", "--universe", str(uni), "--report", str(out)])
    assert res.exit_code == 0, res.output
    md = out.read_text()
    section = md.split("## Underpriced")[1]
    assert "- AAA (score" not in section
    assert "valuation skipped" not in md  # AAA has fundamentals


def test_cli_narrate_without_api_key_exits_with_clear_message(monkeypatch, universe_file):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    calls = []
    monkeypatch.setattr(cli, "fetch", lambda *a, **k: calls.append(a))
    res = runner.invoke(cli.app, ["AAA", "--narrate", "--universe", universe_file])
    assert res.exit_code == 2
    assert "ANTHROPIC_API_KEY" in res.output
    assert calls == []  # no network work before the check


def test_narrate_without_anthropic_package_exits_2(monkeypatch, universe_file):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setitem(__import__("sys").modules, "anthropic", None)  # import -> ImportError
    res = runner.invoke(cli.app, ["AAA", "--narrate", "--universe", universe_file])
    assert res.exit_code == 2
    assert "anthropic" in res.output


def test_narrate_error_warns_and_continues(monkeypatch, tmp_path, universe_file):
    monkeypatch.setattr(cli, "fetch", _fake_fetch())
    monkeypatch.setattr(cli, "_make_client", lambda: object())

    def boom(outlook, facts, client):
        if outlook.ticker == "BBB":
            raise NarrationError("narration failed for BBB: boom")
        return "A short narrative."

    monkeypatch.setattr(cli, "narrate_outlook", boom)
    out = tmp_path / "r.md"
    res = runner.invoke(
        cli.app, ["AAA", "BBB", "--narrate", "--universe", universe_file, "--report", str(out)]
    )
    assert res.exit_code == 0, res.output
    md = out.read_text()
    assert "A short narrative." in md
    assert "narration failed for BBB" in md


def test_narrate_facts_contain_only_that_ticker(monkeypatch, universe_file):
    monkeypatch.setattr(cli, "fetch", _fake_fetch())
    monkeypatch.setattr(cli, "_make_client", lambda: object())
    seen = []

    def spy(outlook, facts, client):
        seen.append((outlook.ticker, facts))
        return "n"

    monkeypatch.setattr(cli, "narrate_outlook", spy)
    res = runner.invoke(cli.app, ["AAA:70", "BBB:30", "--narrate", "--universe", universe_file])
    assert res.exit_code == 0, res.output
    for ticker, facts in seen:
        text = str(facts)
        assert facts["ticker"] == ticker
        assert "weight" not in text
        assert ("BBB" if ticker == "AAA" else "AAA") not in text


def _fund(sector, pe):
    return TickerData("X", "equity", pd.DataFrame(), {"sector": sector, "trailingPE": pe})


def test_sector_median_pe_median_of_same_sector():
    u = {f"T{i}": _fund("Tech", pe) for i, pe in enumerate([10, 20, 30, 40])}
    u["O"] = _fund("Energy", 5)
    assert cli.sector_median_pe(u, "Tech") == 25.0


def test_sector_median_pe_ignores_non_positive_and_missing():
    u = {
        "A": _fund("Tech", 10), "B": _fund("Tech", 30), "C": _fund("Tech", 20),
        "D": _fund("Tech", -5), "E": _fund("Tech", None), "F": _fund("Tech", float("nan")),
    }
    assert cli.sector_median_pe(u, "Tech") == 20.0


def test_sector_median_pe_none_when_few_peers_or_unknown_sector():
    u = {"A": _fund("Tech", 10), "B": _fund("Tech", 30)}
    assert cli.sector_median_pe(u, "Tech") is None
    u["C"] = _fund("Tech", 20)
    assert cli.sector_median_pe(u, None) is None
    assert cli.sector_median_pe(u, "") is None
    assert cli.sector_median_pe(u, "Energy") is None


def test_fetch_universe_reports_progress_and_tolerates_failures():
    seen = []

    def fn(t):
        if t == "BAD":
            raise DataError("x")
        return _td(t)

    out = cli.fetch_universe(["A", "BAD", "B"], fn, lambda d, n: seen.append((d, n)), workers=2)
    assert set(out) == {"A", "B"}
    assert sorted(seen) == [(1, 3), (2, 3), (3, 3)]


def test_fetch_universe_interrupt_cancels_and_reraises():
    def fn(t):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        cli.fetch_universe(["A", "B", "C"], fn, workers=1)


def test_cli_interrupt_during_universe_exits_130(monkeypatch, universe_file):
    base = _fake_fetch()

    def fake(ticker, **kw):
        if ticker.startswith("U"):
            raise KeyboardInterrupt
        return base(ticker)

    monkeypatch.setattr(cli, "fetch", fake)
    res = runner.invoke(cli.app, ["AAA", "--universe", universe_file])
    assert res.exit_code == 130
    assert "Interrupted" in res.output
    assert "Traceback" not in res.output


def test_no_holdings_analysed_skips_universe_fetch(monkeypatch, universe_file):
    calls = []

    def fake(ticker, **kw):
        calls.append(ticker)
        raise DataError(f"no data {ticker}")

    monkeypatch.setattr(cli, "fetch", fake)
    res = runner.invoke(cli.app, ["AAA", "BBB", "--universe", universe_file])
    assert res.exit_code == 1
    assert "No holdings could be analysed" in res.output
    assert "no data AAA" in res.output
    assert not any(t.startswith("U") for t in calls)


def test_missing_universe_file_is_usage_error(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cli, "fetch", lambda *a, **k: calls.append(a))
    res = runner.invoke(cli.app, ["AAA", "--universe", str(tmp_path / "nope.txt")])
    assert res.exit_code == 2
    assert "universe file not found" in res.output
    assert calls == []


def test_bad_report_path_still_prints_report_and_exits_1(monkeypatch, tmp_path, universe_file):
    monkeypatch.setattr(cli, "fetch", _fake_fetch())
    bad = tmp_path / "missing_dir" / "r.md"
    res = runner.invoke(cli.app, ["AAA", "--universe", universe_file, "--report", str(bad)])
    assert res.exit_code == 1
    assert "could not write report" in res.output
    assert "Portfolio analysis" in res.output


def _uni_fetch(skip_fundamentals=()):
    f = {"trailingPE": 8.0, "freeCashflow": 1e9, "marketCap": 1e10, "sector": "Tech"}

    def fake(ticker, **kw):
        if ticker.startswith("U"):
            return _td(ticker, "equity", {} if ticker in skip_fundamentals else dict(f))
        return _td(ticker, "equity")

    return fake


def test_cli_prints_universe_stats_line(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "fetch", _uni_fetch(skip_fundamentals={"U4"}))
    uni = tmp_path / "u.txt"
    uni.write_text("U1\nU2\nU3\nU4\n")
    out = tmp_path / "r.md"
    res = runner.invoke(cli.app, ["AAA", "--universe", str(uni), "--report", str(out)])
    assert res.exit_code == 0, res.output
    md = out.read_text()
    assert "Universe: 4 equities fetched; 1 skipped (missing fundamentals);" in md
    assert "(n=3)" in md
    assert "WARNING" not in md


def test_cli_warns_when_fundamentals_largely_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "fetch", _uni_fetch(skip_fundamentals={"U1", "U2", "U3"}))
    uni = tmp_path / "u.txt"
    uni.write_text("U1\nU2\nU3\nU4\n")
    out = tmp_path / "r.md"
    res = runner.invoke(cli.app, ["AAA", "--universe", str(uni), "--report", str(out)])
    assert res.exit_code == 0, res.output
    md = out.read_text()
    assert "fundamentals were largely unavailable" in md
    assert "No candidates found" in md


@pytest.mark.parametrize("spec", ["none", "NONE", " None "])
def test_universe_none_skips_screen_entirely(monkeypatch, tmp_path, spec):
    fetched: list[str] = []
    base = _fake_fetch()

    def fake(ticker, **kw):
        fetched.append(ticker)
        return base(ticker, **kw)

    def boom(_spec):
        raise AssertionError("load_universe must not be called")

    monkeypatch.setattr(cli, "fetch", fake)
    monkeypatch.setattr(cli, "load_universe", boom)
    out = tmp_path / "r.md"
    res = runner.invoke(cli.app, ["AAA", "--universe", spec, "--report", str(out)])
    assert res.exit_code == 0, res.output
    assert "Underpriced screen skipped (--universe none)" in out.read_text()
    assert sorted(fetched) == ["AAA", "SPY"]  # holdings + benchmark only
