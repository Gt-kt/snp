import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from titan.alpaca_executor import AlpacaExecutor
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Titan Trade Dashboard")

# Serve the static files (HTML, CSS, JS) from the 'static' directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_dashboard():
    return FileResponse("static/index.html")

@app.get("/api/scan-results")
async def get_scan_results():
    """Returns the latest scan data saved by titan_trade_v3.py"""
    try:
        with open("data/latest_scan.json", "r") as f:
            data = json.load(f)
        return {"status": "success", "data": data}
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

        # Fetch basic account info
        account_info = {
            "buying_power": executor.get_buying_power(),
            "cash": float(executor.trading_client.get_account().cash),
            "portfolio_value": float(executor.trading_client.get_account().portfolio_value),
            "day_trade_count": executor.trading_client.get_account().daytrade_count
        }

        # Fetch open positions
        positions = executor.trading_client.get_all_positions()
        active_positions = []
        for p in positions:
            active_positions.append({
                "symbol": p.symbol,
                "qty": float(p.qty),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc) * 100, # Convert to percentage
                "current_price": float(p.current_price),
                "avg_entry_price": float(p.avg_entry_price)
            })

        return {
            "status": "success",
            "account": account_info,
            "positions": active_positions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alpaca API Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dashboard_app:app", host="127.0.0.1", port=8000, reload=True)
