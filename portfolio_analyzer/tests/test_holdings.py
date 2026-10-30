import pytest

from portfolio_analyzer.holdings import parse_holdings


def test_weights_normalised():
    holdings, warnings = parse_holdings(["AAPL:40", "MSFT:30"])
    assert [h.ticker for h in holdings] == ["AAPL", "MSFT"]
    assert holdings[0].weight == pytest.approx(4 / 7)
    assert holdings[1].weight == pytest.approx(3 / 7)
    assert len(warnings) == 1


def test_equal_weights_when_omitted():
    holdings, warnings = parse_holdings(["AAPL", "BTC-USD"])
    assert [h.weight for h in holdings] == [0.5, 0.5]
    assert warnings == []


def test_mixed_weighted_unweighted_raises():
    with pytest.raises(ValueError, match="all or none"):
        parse_holdings(["AAPL:50", "MSFT"])


def test_uppercases():
    holdings, _ = parse_holdings(["aapl:50", "cl=f:50"])
    assert [h.ticker for h in holdings] == ["AAPL", "CL=F"]


def test_duplicate_raises():
    with pytest.raises(ValueError, match="AAPL"):
        parse_holdings(["AAPL:50", "aapl:50"])


def test_futures_and_crypto_symbols_parse():
    holdings, _ = parse_holdings(["CL=F:5", "BTC-USD:10"])
    assert [h.ticker for h in holdings] == ["CL=F", "BTC-USD"]
    assert holdings[0].weight == pytest.approx(1 / 3)


@pytest.mark.parametrize("bad", ["AAPL:abc", "AAPL:-5", "AAPL:0"])
def test_bad_weight_raises(bad):
    with pytest.raises(ValueError):
        parse_holdings([bad])


def test_empty_raises():
    with pytest.raises(ValueError):
        parse_holdings([])
