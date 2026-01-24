
from titan_trade import TitanBrain, tabulate
from datetime import datetime

def run_auto_scan():
    print("Starting Automated Titan Scan...")
    brain = TitanBrain()
    try:
        setups, stats = brain.scan()
        
        if setups:
            setups.sort(key=lambda x: x.score, reverse=True)
            
            print("\n" * 3)
            print("="*60)
            print(f"  TITAN GUARDIAN v6.0 - RESULT SUMMARY (AUTO)")
            print("="*60)
            
            table = []
            for s in setups[:20]: # Show Top 20
                dist = (s.trigger - s.price) / s.price
                status = "🚀 BUY NOW" if s.price >= s.trigger else ("⚠️ NEAR" if dist < 0.01 else "⏳ PENDING")

                table.append([
                    s.ticker, 
                    s.strategy[:4], 
                    f"${s.price:.2f}",
                    f"${s.trigger:.2f}", 
                    status,
                    f"{s.win_rate:.0f}%", 
                    f"{s.profit_factor:.2f}",
                    f"${s.target:.2f}",
                    s.note
                ])
                
            print(tabulate(table, headers=["Ticker", "Type", "Price", "Trig", "Status", "WR%", "PF", "Target", "Note"], tablefmt="fancy_grid"))
            
            print("\nSCAN STATISTICS:")
            for k, v in stats.items():
                print(f"{k}: {v}")
                
        else:
            print("No setups found.")
            
    except Exception as e:
        print(f"Scan Error: {e}")

if __name__ == "__main__":
    run_auto_scan()
