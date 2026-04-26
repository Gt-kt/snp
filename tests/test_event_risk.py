import titan.event_risk as er


def test_event_risk_high_for_earnings_window():
    out = er.assess_event_risk("AAA", earnings_days=2)
    assert out["status"] == "HIGH"
    assert "earnings" in out["reasons"][0]


def test_event_risk_medium_for_analyst_news(monkeypatch):
    class FakeTicker:
        news = [{"title": "Analyst raises price target on AAA"}]

    class FakeYF:
        @staticmethod
        def Ticker(ticker):
            return FakeTicker()

    monkeypatch.setitem(__import__("sys").modules, "yfinance", FakeYF)
    out = er.assess_event_risk("AAA", earnings_days=30)
    assert out["status"] == "MED"


def test_event_risk_high_for_bad_headline(monkeypatch):
    class FakeTicker:
        news = [{"title": "AAA shares fall after SEC investigation"}]

    class FakeYF:
        @staticmethod
        def Ticker(ticker):
            return FakeTicker()

    monkeypatch.setitem(__import__("sys").modules, "yfinance", FakeYF)
    out = er.assess_event_risk("AAA", earnings_days=30)
    assert out["status"] == "HIGH"


def test_event_risk_does_not_match_sec_inside_sector(monkeypatch):
    class FakeTicker:
        news = [{"title": "Morgan Stanley adjusts valuation amid utility sector strength"}]

    class FakeYF:
        @staticmethod
        def Ticker(ticker):
            return FakeTicker()

    monkeypatch.setitem(__import__("sys").modules, "yfinance", FakeYF)
    out = er.assess_event_risk("AAA", earnings_days=30)
    assert out["status"] == "LOW"


def test_event_risk_earnings_headline_is_medium_without_date_blackout(monkeypatch):
    class FakeTicker:
        news = [{"title": "Linde earnings expected to grow"}]

    class FakeYF:
        @staticmethod
        def Ticker(ticker):
            return FakeTicker()

    monkeypatch.setitem(__import__("sys").modules, "yfinance", FakeYF)
    out = er.assess_event_risk("AAA", earnings_days=30)
    assert out["status"] == "MED"


def test_event_risk_low_without_recent_headlines(monkeypatch):
    class FakeTicker:
        news = []

    class FakeYF:
        @staticmethod
        def Ticker(ticker):
            return FakeTicker()

    monkeypatch.setitem(__import__("sys").modules, "yfinance", FakeYF)
    out = er.assess_event_risk("AAA", earnings_days=30)
    assert out["status"] == "LOW"
