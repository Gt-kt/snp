"""
Titan Trade Alpaca Executor
===========================
Handles connection and order execution using the Alpaca Trade API.
"""

import os
import logging
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame

logger = logging.getLogger("titan")

class AlpacaExecutor:
    """Manages Alpaca API connection and simulated/live order routing."""
    
    def __init__(self, api_key=None, secret_key=None, use_paper=True):
        self.api_key = api_key or os.environ.get("APCA_API_KEY_ID")
        self.secret_key = secret_key or os.environ.get("APCA_API_SECRET_KEY")
        self.base_url = "https://paper-api.alpaca.markets" if use_paper else "https://api.alpaca.markets"
        
        self.api = None
        self.connected = False
        
        if not self.api_key or not self.secret_key:
            logger.warning("Alpaca API keys not found in environment variables.")
            return
            
        try:
            self.api = tradeapi.REST(
                key_id=self.api_key,
                secret_key=self.secret_key,
                base_url=self.base_url,
                api_version='v2'
            )
            self.account = self.api.get_account()
            if self.account.status == 'ACTIVE':
                self.connected = True
                logger.info(f"Connected to Alpaca! Buying Power: ${float(self.account.buying_power):,.2f}")
            else:
                logger.warning(f"Alpaca account status is {self.account.status}")
                
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca API: {e}")

    def is_connected(self):
        return self.connected
        
    def get_buying_power(self):
        if not self.connected:
            return 0.0
        try:
            self.account = self.api.get_account()
            return float(self.account.buying_power)
        except Exception:
            return 0.0

    def get_account_equity(self):
        """Return current account equity, or 0 on failure."""
        if not self.connected:
            return 0.0
        try:
            self.account = self.api.get_account()
            return float(self.account.equity)
        except Exception:
            return 0.0


    def get_account_last_equity(self):
        """Return prior-close account equity, or 0 on failure."""
        if not self.connected:
            return 0.0
        try:
            self.account = self.api.get_account()
            return float(getattr(self.account, 'last_equity', 0.0) or 0.0)
        except Exception:
            return 0.0
    def get_open_order_symbols(self):
        """Return a list of symbols that already have open orders."""
        if not self.connected:
            return []
        try:
            return [o.symbol for o in self.api.list_orders(status='open')]
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    def get_open_positions_snapshot(self):
        """Return open positions keyed by symbol for portfolio-level risk checks."""
        if not self.connected:
            return {}
        snapshot = {}
        try:
            for pos in self.api.list_positions():
                qty = abs(int(float(pos.qty)))
                entry_price = float(pos.avg_entry_price)
                snapshot[pos.symbol] = {
                    'entry_price': entry_price,
                    'shares': qty,
                    'market_value': abs(float(getattr(pos, 'market_value', entry_price * qty))),
                }
            return snapshot
        except Exception as e:
            logger.error(f"Failed to get position snapshot: {e}")
            return {}
            
    def get_open_orders(self, symbol=None, side=None):
        """Return normalized open orders, optionally filtered by symbol/side."""
        if not self.connected:
            return []
        orders_out = []
        try:
            orders = self.api.list_orders(status='open', nested=False)
            for order in orders:
                if symbol and order.symbol != symbol:
                    continue
                if side and order.side != side:
                    continue
                qty = 0
                try:
                    qty = abs(int(float(getattr(order, 'qty', 0) or 0)))
                except Exception:
                    pass
                limit_price = getattr(order, 'limit_price', None)
                stop_price = getattr(order, 'stop_price', None)
                orders_out.append({
                    'id': order.id,
                    'symbol': order.symbol,
                    'side': order.side,
                    'type': getattr(order, 'type', ''),
                    'status': getattr(order, 'status', ''),
                    'qty': qty,
                    'limit_price': float(limit_price) if limit_price not in (None, '') else None,
                    'stop_price': float(stop_price) if stop_price not in (None, '') else None,
                })
            return orders_out
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    def cancel_orders_for_symbol(self, symbol, side=None, order_type=None):
        """Cancel matching open orders for a symbol."""
        if not self.connected:
            return 0
        cancelled = 0
        try:
            for order in self.get_open_orders(symbol=symbol, side=side):
                if order_type and order.get('type') != order_type:
                    continue
                self.api.cancel_order(order['id'])
                cancelled += 1
            return cancelled
        except Exception as e:
            logger.error(f"Failed to cancel orders for {symbol}: {e}")
            return cancelled

    def submit_limit_order(self, symbol, qty, side, limit_price, time_in_force='day', extended_hours=False):
        """Submit a plain limit order."""
        if not self.connected:
            logger.error("Cannot submit limit order: Not connected to Alpaca.")
            return False
        try:
            self.api.submit_order(
                symbol=symbol,
                qty=int(qty),
                side=side,
                type='limit',
                time_in_force=time_in_force,
                limit_price=round(limit_price, 2),
                extended_hours=extended_hours,
            )
            logger.info(f"Submitted {side} limit order for {qty} {symbol} @ ${limit_price:.2f}")
            return True
        except Exception as e:
            logger.error(f"Failed limit order for {symbol}: {e}")
            return False

    def submit_market_order(self, symbol, qty, side, time_in_force='day'):
        """Submit a plain market order."""
        if not self.connected:
            logger.error("Cannot submit market order: Not connected to Alpaca.")
            return False
        try:
            self.api.submit_order(
                symbol=symbol,
                qty=int(qty),
                side=side,
                type='market',
                time_in_force=time_in_force,
            )
            logger.info(f"Submitted {side} market order for {qty} {symbol}")
            return True
        except Exception as e:
            logger.error(f"Failed market order for {symbol}: {e}")
            return False

    def submit_stop_order(self, symbol, qty, stop_price, time_in_force='gtc'):
        """Submit a protective stop sell order."""
        if not self.connected:
            logger.error("Cannot submit stop order: Not connected to Alpaca.")
            return False
        try:
            self.api.submit_order(
                symbol=symbol,
                qty=int(qty),
                side='sell',
                type='stop',
                time_in_force=time_in_force,
                stop_price=round(stop_price, 2),
            )
            logger.info(f"Submitted stop order for {qty} {symbol} @ stop ${stop_price:.2f}")
            return True
        except Exception as e:
            logger.error(f"Failed stop order for {symbol}: {e}")
            return False

    def get_open_positions_count(self):
        """Returns the number of current open positions."""
        if not self.connected:
            return 0
        try:
            positions = self.api.list_positions()
            return len(positions)
        except Exception as e:
            logger.error(f"Failed to get open positions: {e}")
            return 0

    def submit_bracket_order(self, symbol, qty, entry_price, target_price, stop_price):
        """
        Submits an institutional-style OCO (One Cancels Other) bracket order.
        This sends the entry limit, and automatically attaches trailing take profit and stop loss.
        """
        if not self.connected:
            logger.error("Cannot submit order: Not connected to Alpaca.")
            return False
            
        try:
            # Check if we already have an open position or pending order for this symbol
            positions = [p.symbol for p in self.api.list_positions()]
            if symbol in positions:
                logger.warning(f"Already hold a position in {symbol}. Skipping redundant order.")
                return False
                
            open_orders = [o.symbol for o in self.api.list_orders(status='open')]
            if symbol in open_orders:
                logger.warning(f"Already have an open order for {symbol}. Skipping redundant order.")
                return False

            # Submit the bracket order!
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side='buy',
                type='limit',
                time_in_force='day',
                limit_price=round(entry_price, 2),
                order_class='bracket',
                take_profit=dict(
                    limit_price=round(target_price, 2)
                ),
                stop_loss=dict(
                    stop_price=round(stop_price, 2)
                )
            )
            logger.info(f"SUCCESS: Submitted bracket order for {qty} shares of {symbol} at Limit ${entry_price:.2f}.")
            return True
            
        except Exception as e:
            logger.error(f"FAILED to submit order for {symbol}: {e}")
            return False


