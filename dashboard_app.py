import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from titan.alpaca_executor import AlpacaExecutor
from titan.opportunity import normalize_scan_payload
from dotenv import load_dotenv

load_dotenv()

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
SCAN_RESULTS_FILE = APP_DIR / "data" / "latest_scan.json"

app = FastAPI(title="Titan Trade Dashboard")

# Serve the static files (HTML, CSS, JS) from the 'static' directory
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def serve_dashboard():
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.get("/api/scan-results")
async def get_scan_results():
    """Returns the latest scan data saved by titan_trade_v3.py"""
    try:
        with SCAN_RESULTS_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {"status": "success", "data": normalize_scan_payload(data)}
    except FileNotFoundError:
        return {"status": "pending", "message": "No scan results found yet. Run 'python titan_trade_v3.py' to generate data."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/portfolio")
async def get_portfolio():
    """Connects to Alpaca to return live portfolio data and open positions"""
    try:
        executor = AlpacaExecutor()
        if not executor.is_connected():
            return {"status": "error", "message": "Failed to connect to Alpaca. Check API keys."}

        account_info = executor.get_account_summary()
        if account_info is None:
            raise HTTPException(status_code=502, detail="Connected to Alpaca but failed to load account data.")

        return {
            "status": "success",
            "account": account_info,
            "positions": executor.get_positions_summary()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alpaca API Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dashboard_app:app", host="127.0.0.1", port=8000, reload=True)
