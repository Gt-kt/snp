# Titan Trade Manual Swing Scanner

Titan Trade is a manual S&P 500 swing-trading scanner for 3-5 day holds.
It is designed to help you find stalk/watch candidates, decide whether a
setup is worth buying, and manage exits. The default workflow does not send
broker orders.

## Safety Model

- Manual scan mode is the default.
- Do not commit `.env`, local position logs, scan outputs, or cache files.
- Live Alpaca routing requires explicit CLI flags and should stay off unless
  you have reviewed the generated levels.
- If you expose the dashboard beyond localhost, set `TITAN_DASHBOARD_API_TOKEN`.
- Rotate any broker keys that were ever committed to Git history.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e ".[dev]"
copy .env.example .env
```

Edit `.env` only if you need Alpaca account data:

```text
APCA_API_KEY_ID=...
APCA_API_SECRET_KEY=...
TITAN_ALPACA_USE_PAPER=true
```

## Manual 3-5 Day Swing Workflow

1. Run the pro scan after the US market opens or near the close.

```powershell
python titan_trade_v3.py --pro
```

After editable install, the same command is available as:

```powershell
titan-scan --pro
```

2. Review the final decision, stalk orders, watchlist, and risk notes.
   Prefer setups with clear entry, stop, target, sector alignment, and no
   earnings/news blocker.

3. Buy manually only if price is inside the planned buy zone. Do not chase
   past the scanner's max-buy guidance.

4. Add the actual fill to the dashboard so the tracker can monitor stop,
   target, trailing stop, and time stop.

```powershell
python titan_dashboard.py
```

After editable install:

```powershell
titan-dashboard
```

Open `http://127.0.0.1:8000`.

On Windows, the simplest launcher is:

```powershell
.\RUN_DASHBOARD.bat
```

By default the dashboard is position-tracking only and reads the latest saved
scan. To make the dashboard run scans automatically on startup and every
interval:

```powershell
python titan_dashboard.py --auto-scan --interval 15
```

5. Sell manually when the dashboard gives a stop, target, trailing-stop, or
   time-stop alert. The intended holding period is short: usually 3-5 trading
   days, not an open-ended position trade.

## Dashboard Security

Local-only dashboard:

```powershell
python titan_dashboard.py --host 127.0.0.1 --port 8000
```

Network-exposed dashboard:

```powershell
$env:TITAN_DASHBOARD_API_TOKEN="use-a-long-random-token"
python titan_dashboard.py --host 0.0.0.0 --port 8000
```

Without `TITAN_DASHBOARD_API_TOKEN`, binding to `0.0.0.0` is rejected unless
you pass `--allow-unsafe-no-auth`.

## Tests

```powershell
pytest -q
```

The suite is expected to run offline with mocked/stubbed market data paths.

## Runtime Files

The app creates local state and cache files such as:

- `.env`
- `data/latest_scan.json`
- `my_positions.json`
- `portfolio.json`
- `signal_log.json`
- `trust_mode_state.json`
- `cache_sp500_*`

These are intentionally ignored by Git.
