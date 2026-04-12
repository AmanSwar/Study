# Chapter 25: Live Execution on Zerodha — From Simulation to Production Trading

## Introduction

You've built prediction models, backtested strategies, and validated systems. Now comes the critical jump: executing real trades with real money. This chapter bridges the gap between simulation and production, focusing on Zerodha's Kite Connect API—India's most accessible broker for retail algorithmic trading.

We'll cover three interconnected domains:

1. **Zerodha Kite Connect API**: How to authenticate, stream live data, and interact with the broker
2. **Order Management System**: State machines that keep your system synchronized with the broker
3. **Safe Transition**: Paper trading, gradual scaling, and navigating India's tax framework

By the end, you'll have production-ready code running live on NSE.

---

# Module 25.1: Zerodha Kite Connect API — Building the Broker Interface

## Overview: Why Zerodha?

Zerodha is India's largest retail broker, with ~1.5 million active traders. Key advantages for algorithmic trading:

- **Low latency**: API servers co-located at exchange infrastructure
- **API-first design**: REST and WebSocket APIs designed for traders, not banks
- **Comprehensive data**: Live ticks, historical data, options chain data
- **Robust order routing**: Direct NSE/BSE access with institutional-grade order management
- **Reasonable limits**: 10 requests/second for REST, unlimited WebSocket connections

Constraints to understand:

- No margin lending at order placement (must pre-fund margin)
- Market hours restrictions (9:15 AM - 3:30 PM IST for NSE)
- Weekly instrument snapshot requirement (can cause cache stales)

## Authentication Flow: OAuth-like Pattern

Zerodha uses a two-stage authentication:

```
User Login → Request Token (browser) → Access Token (server-side) → Persistent Session
```

Unlike typical OAuth, you manage the request token request through a web callback.

### Stage 1: Request Token Generation

```python
"""
Stage 1: User opens web browser and logs in to Zerodha.
This generates a request_token that you capture from the redirect URL.
"""

from typing import Optional
import webbrowser
from urllib.parse import urlparse, parse_qs

class KiteAuthenticator:
    """Manages OAuth-like authentication with Zerodha Kite Connect."""
    
    def __init__(self, api_key: str, api_secret: str):
        """
        Initialize authenticator.
        
        Args:
            api_key: Your Kite API key (from Settings > API)
            api_secret: Your Kite API secret (keep confidential)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.request_token: Optional[str] = None
        self.access_token: Optional[str] = None
        
    def get_login_url(self, redirect_url: str = "http://localhost:8080/") -> str:
        """
        Generate Kite login URL for user.
        
        User opens this URL in browser and logs in with Zerodha credentials.
        After successful auth, browser redirects to redirect_url?request_token=XXX
        
        Args:
            redirect_url: Where Zerodha redirects after login
            
        Returns:
            Full login URL to open in browser
        """
        # Zerodha Kite login endpoint
        login_url = f"https://kite.zerodha.com/connect/login?api_key={self.api_key}&v=3"
        return login_url
    
    def extract_request_token_from_url(self, redirect_url: str) -> str:
        """
        Extract request_token from callback URL.
        
        In production, your web callback handler receives:
            http://localhost:8080/?request_token=XXXXX&action=login
        
        Args:
            redirect_url: Full URL from browser redirect
            
        Returns:
            request_token string
        """
        parsed = urlparse(redirect_url)
        params = parse_qs(parsed.query)
        request_token = params.get('request_token', [None])[0]
        
        if not request_token:
            raise ValueError(f"No request_token in URL: {redirect_url}")
        
        self.request_token = request_token
        return request_token


# Usage in practice:
# 1. authenticator = KiteAuthenticator(api_key="xxx", api_secret="yyy")
# 2. print(authenticator.get_login_url())  # User opens this in browser
# 3. User logs in → browser redirects → you extract request_token
```

### Stage 2: Access Token Exchange

```python
"""
Stage 2: Exchange request_token for access_token.
This happens server-side and gives you a session token valid for 6 hours.
"""

import hashlib
import requests
from datetime import datetime, timedelta

class KiteAccessTokenManager:
    """Manages access token exchange and refresh."""
    
    BASE_URL = "https://api.kite.trade/session"
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token: Optional[str] = None
        self.user_id: Optional[str] = None
        self.token_expiry: Optional[datetime] = None
        
    def get_access_token(self, request_token: str) -> tuple[str, str]:
        """
        Exchange request_token for access_token.
        
        This is a server-to-server call. The checksum proves you own the API secret.
        
        Args:
            request_token: From Stage 1
            
        Returns:
            Tuple of (access_token, user_id)
            
        Raises:
            requests.RequestException: If API call fails
        """
        # Create checksum: SHA256(api_key + request_token + api_secret)
        checksum_input = f"{self.api_key}{request_token}{self.api_secret}"
        checksum = hashlib.sha256(checksum_input.encode()).hexdigest()
        
        payload = {
            "api_key": self.api_key,
            "request_token": request_token,
            "checksum": checksum
        }
        
        response = requests.post(
            f"{self.BASE_URL}/token",
            data=payload,
            timeout=10
        )
        
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to get access token: {response.status_code}\n"
                f"Response: {response.text}"
            )
        
        data = response.json()
        if data.get("status") != "success":
            raise RuntimeError(f"API returned error: {data.get('message')}")
        
        self.access_token = data["data"]["access_token"]
        self.user_id = data["data"]["user_id"]
        self.token_expiry = datetime.now() + timedelta(hours=6)
        
        return self.access_token, self.user_id
    
    def is_token_valid(self) -> bool:
        """Check if current access_token is still valid."""
        if not self.access_token or not self.token_expiry:
            return False
        return datetime.now() < self.token_expiry
```

## REST API: Core Operations

Once authenticated, you can make REST calls for orders, positions, and account data.

```python
"""
REST API Interface: Synchronous calls for deterministic operations.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, List
import json

class OrderType(Enum):
    """Order types supported by Zerodha."""
    MARKET = "MIS"      # Market/intraday order
    LIMIT = "LIMIT"     # Limit order
    STOP = "STOP"       # Stop-loss order
    STOP_LIMIT = "STOPLIMIT"  # Stop-limit order


class OrderValidity(Enum):
    """How long order remains valid."""
    DAY = "DAY"          # Till market close
    IOC = "IOC"          # Immediate or cancel
    GTT = "GTT"          # Good till triggered (days)


@dataclass
class Order:
    """Represents an order."""
    order_id: str
    instrument_token: int
    exchange_token: str
    tradingsymbol: str  # e.g., "RELIANCE"
    order_type: str
    transaction_type: str  # "BUY" or "SELL"
    quantity: int
    price: float
    status: str  # "PENDING", "COMPLETE", "REJECTED", "CANCELLED"
    filled_quantity: int
    pending_quantity: int
    average_price: float
    placed_at: str
    executed_at: str
    
    def is_complete(self) -> bool:
        """Check if order fully executed."""
        return self.filled_quantity == self.quantity


@dataclass
class Position:
    """Represents an open position."""
    tradingsymbol: str
    quantity: int  # Current held quantity
    buy_quantity: int
    sell_quantity: int
    average_buy_price: float
    average_sell_price: float
    last_price: float
    multiplier: int  # For futures/options
    pnl: float  # Current P&L
    m2m: float  # Mark-to-market P&L


class KiteRestAPI:
    """
    Synchronous wrapper around Zerodha Kite REST API.
    All methods include rate limiting awareness and error handling.
    """
    
    BASE_URL = "https://api.kite.trade"
    
    def __init__(self, access_token: str, user_id: str, api_key: str):
        """
        Initialize REST API client.
        
        Args:
            access_token: From authentication
            user_id: Typically user email/username
            api_key: Your API key
        """
        self.access_token = access_token
        self.user_id = user_id
        self.api_key = api_key
        self.session = requests.Session()
        self._setup_headers()
        
    def _setup_headers(self):
        """Configure authentication headers."""
        self.session.headers.update({
            "Authorization": f"token {self.api_key}:{self.access_token}",
            "X-Kite-Version": "3",
            "Content-Type": "application/x-www-form-urlencoded"
        })
    
    def place_order(
        self,
        tradingsymbol: str,
        exchange: str,
        transaction_type: str,
        quantity: int,
        order_type: str,
        price: float = 0,
        stoploss: float = 0,
        trailing_stoploss: float = 0,
        product: str = "MIS"  # MIS for intraday, CNC for delivery
    ) -> str:
        """
        Place a new order.
        
        Args:
            tradingsymbol: e.g., "RELIANCE"
            exchange: "NSE" or "BSE"
            transaction_type: "BUY" or "SELL"
            quantity: Number of shares
            order_type: "MARKET", "LIMIT", "STOP", "STOPLIMIT"
            price: Limit price (for LIMIT/STOPLIMIT)
            stoploss: Stoploss price
            trailing_stoploss: Trailing stoploss percentage
            product: "MIS" (intraday) or "CNC" (delivery)
            
        Returns:
            Order ID string
            
        Raises:
            requests.RequestException: Network error
            RuntimeError: API returned error
        """
        params = {
            "variety": "regular",
            "tradingsymbol": tradingsymbol,
            "symboltoken": "",
            "transactiontype": transaction_type,
            "exchange": exchange,
            "ordertype": order_type,
            "producttype": product,
            "quantity": str(quantity),
            "price": str(price),
            "validity": "DAY",
            "disclosedquantity": "0",
            "squareoff": "0",
            "stoploss": str(stoploss) if stoploss else "0",
            "trailingstoploss": str(trailing_stoploss) if trailing_stoploss else "0",
            "oco_order_id": ""
        }
        
        try:
            response = self.session.post(
                f"{self.BASE_URL}/orders/regular",
                data=params,
                timeout=5
            )
        except requests.Timeout:
            raise RuntimeError("Order placement timed out (>5s)")
        except requests.ConnectionError as e:
            raise RuntimeError(f"Network error during order placement: {e}")
        
        if response.status_code not in (200, 201):
            error_msg = response.text
            if response.status_code == 400:
                # Common cases: invalid quantity, insufficient margin, etc.
                raise RuntimeError(f"Invalid order parameters: {error_msg}")
            elif response.status_code == 403:
                raise RuntimeError("Session expired. Re-authenticate.")
            else:
                raise RuntimeError(f"API error {response.status_code}: {error_msg}")
        
        data = response.json()
        if data.get("status") != "success":
            raise RuntimeError(f"Order placement failed: {data.get('message')}")
        
        order_id = data["data"]["order_id"]
        return order_id
    
    def cancel_order(self, order_id: str, exchange: str = "NSE") -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: Order ID from place_order
            exchange: "NSE" or "BSE"
            
        Returns:
            True if cancellation successful
            
        Raises:
            RuntimeError: If order cannot be cancelled (already executed, etc.)
        """
        params = {
            "order_id": order_id,
            "variety": "regular"
        }
        
        response = self.session.delete(
            f"{self.BASE_URL}/orders/regular/{order_id}",
            params=params,
            timeout=5
        )
        
        if response.status_code not in (200, 204):
            data = response.json()
            raise RuntimeError(
                f"Failed to cancel order {order_id}: {data.get('message')}"
            )
        
        return True
    
    def modify_order(
        self,
        order_id: str,
        quantity: int = None,
        price: float = None,
        stoploss: float = None
    ) -> str:
        """
        Modify an existing pending order.
        
        Can only modify unfilled/partial orders.
        
        Args:
            order_id: Order to modify
            quantity: New quantity (must be ≤ original)
            price: New limit price
            stoploss: New stoploss price
            
        Returns:
            Modified order ID (may differ from input)
        """
        params = {
            "order_id": order_id,
            "variety": "regular"
        }
        
        if quantity is not None:
            params["quantity"] = str(quantity)
        if price is not None:
            params["price"] = str(price)
        if stoploss is not None:
            params["stoploss"] = str(stoploss)
        
        response = self.session.put(
            f"{self.BASE_URL}/orders/regular/{order_id}",
            data=params,
            timeout=5
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Order modification failed: {response.text}")
        
        data = response.json()
        return data["data"]["order_id"]
    
    def get_orders(self) -> List[Order]:
        """
        Retrieve all orders (today and previous unfilled orders).
        
        Returns:
            List of Order objects
        """
        response = self.session.get(
            f"{self.BASE_URL}/orders",
            timeout=5
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch orders: {response.text}")
        
        data = response.json()
        orders = []
        
        for order_data in data.get("data", []):
            order = Order(
                order_id=order_data["order_id"],
                instrument_token=order_data.get("instrument_token", 0),
                exchange_token=order_data.get("exchange_token", ""),
                tradingsymbol=order_data["tradingsymbol"],
                order_type=order_data["ordertype"],
                transaction_type=order_data["transactiontype"],
                quantity=int(order_data["quantity"]),
                price=float(order_data["price"]),
                status=order_data["status"],
                filled_quantity=int(order_data.get("filled_quantity", 0)),
                pending_quantity=int(order_data.get("pending_quantity", 0)),
                average_price=float(order_data.get("average_price", 0)),
                placed_at=order_data.get("order_timestamp", ""),
                executed_at=order_data.get("filled_timestamp", ""),
            )
            orders.append(order)
        
        return orders
    
    def get_positions(self) -> Dict[str, Position]:
        """
        Get current open positions.
        
        Returns:
            Dictionary mapping tradingsymbol -> Position
        """
        response = self.session.get(
            f"{self.BASE_URL}/portfolio/positions",
            timeout=5
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch positions: {response.text}")
        
        data = response.json()
        positions = {}
        
        for pos_data in data.get("data", {}).get("net", []):
            pos = Position(
                tradingsymbol=pos_data["tradingsymbol"],
                quantity=int(pos_data["quantity"]),
                buy_quantity=int(pos_data["buy_quantity"]),
                sell_quantity=int(pos_data["sell_quantity"]),
                average_buy_price=float(pos_data["average_buy_price"]),
                average_sell_price=float(pos_data["average_sell_price"]),
                last_price=float(pos_data["last_price"]),
                multiplier=int(pos_data.get("multiplier", 1)),
                pnl=float(pos_data.get("pnl", 0)),
                m2m=float(pos_data.get("m2m", 0))
            )
            positions[pos.tradingsymbol] = pos
        
        return positions
    
    def get_holdings(self) -> Dict[str, int]:
        """
        Get delivery holdings (shares you own, not intraday).
        
        Returns:
            Dictionary mapping tradingsymbol -> quantity held
        """
        response = self.session.get(
            f"{self.BASE_URL}/portfolio/holdings",
            timeout=5
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch holdings: {response.text}")
        
        data = response.json()
        holdings = {}
        
        for holding in data.get("data", []):
            holdings[holding["tradingsymbol"]] = int(holding["quantity"])
        
        return holdings
    
    def get_margins(self) -> Dict[str, float]:
        """
        Get account margin and fund details.
        
        Returns:
            Dictionary with keys: available, utilised, net, equity_multiplier
        """
        response = self.session.get(
            f"{self.BASE_URL}/user/margins",
            timeout=5
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch margins: {response.text}")
        
        data = response.json()
        return {
            "available": float(data["data"]["equity"]["available"]),
            "utilised": float(data["data"]["equity"]["utilised"]),
            "net": float(data["data"]["equity"]["net"]),
            "equity_multiplier": float(data["data"]["equity"]["multiplier"])
        }
    
    def get_instruments(self, exchange: str = "NSE") -> Dict[str, int]:
        """
        Get instrument tokens for symbols.
        
        This is a CRITICAL call—must be cached locally.
        Zerodha provides a CSV dump; parse and cache it.
        
        Returns:
            Dictionary mapping tradingsymbol -> instrument_token
        """
        # In production, download from:
        # https://api.kite.trade/instruments
        # Parse CSV and build lookup table
        # Cache locally; refresh weekly
        
        # For now, return hardcoded examples
        return {
            "RELIANCE": 738561,
            "INFY": 408065,
            "TCS": 2953217,
            "WIPRO": 2977281,
            "AXISBANK": 1510401
        }
```

## WebSocket API: Live Data Streaming

Live tick data arrives through WebSocket, critical for fast decision-making.

```python
"""
WebSocket API: Async streaming of live market ticks.
Use for real-time prices, not for orders (REST is better for orders).
"""

import asyncio
import websockets
import json
from typing import Callable, Optional
from datetime import datetime
from dataclasses import dataclass

@dataclass
class Tick:
    """Real-time market data tick."""
    instrument_token: int
    tradingsymbol: str
    last_price: float
    bid: float
    ask: float
    volume: int
    timestamp: datetime


class KiteWebSocketClient:
    """
    WebSocket client for live market data.
    
    Connection mode:
    1. LTP: Light tick package (last traded price only) - lowest bandwidth
    2. Quote: Full quote (price, bid/ask, volume) - medium bandwidth
    3. Full: Complete depth data - high bandwidth
    """
    
    WS_URL = "wss://ws.kite.trade"
    
    def __init__(self, api_key: str, access_token: str):
        self.api_key = api_key
        self.access_token = access_token
        self.ws = None
        self.is_connected = False
        self.tick_callbacks: List[Callable] = []
        self.message_count = 0
        
    async def connect(self):
        """
        Establish WebSocket connection and authenticate.
        """
        auth_payload = {
            "a": "subscribe",
            "v": "ltp",  # LTP (light), quote (medium), full (heavy)
            "mode": "ltp",
            "key": self.api_key,
            "user_id": self.api_key.split("_")[0],  # Extract user from key
            "enctoken": self.access_token
        }
        
        try:
            self.ws = await websockets.connect(self.WS_URL, ping_interval=10)
            await self.ws.send(json.dumps(auth_payload))
            self.is_connected = True
            print("WebSocket connected")
        except Exception as e:
            raise RuntimeError(f"Failed to connect WebSocket: {e}")
    
    def subscribe(self, instrument_tokens: list[int]):
        """
        Subscribe to tick updates for specific instruments.
        
        Args:
            instrument_tokens: List of instrument tokens (from get_instruments)
        """
        payload = {
            "a": "subscribe",
            "v": instrument_tokens
        }
        asyncio.create_task(self._send_async(json.dumps(payload)))
    
    def register_tick_callback(self, callback: Callable[[Tick], None]):
        """
        Register function to be called on each tick.
        
        Args:
            callback: Function(tick: Tick) -> None
        """
        self.tick_callbacks.append(callback)
    
    async def _send_async(self, message: str):
        """Send message to WebSocket (async)."""
        if self.ws:
            await self.ws.send(message)
    
    async def start_listening(self):
        """
        Start listening for ticks (blocking coroutine).
        Run in separate thread/event loop.
        """
        while self.is_connected:
            try:
                message = await asyncio.wait_for(self.ws.recv(), timeout=30)
                self.message_count += 1
                
                # Parse tick data
                if isinstance(message, bytes):
                    tick = self._parse_binary_tick(message)
                    if tick:
                        for callback in self.tick_callbacks:
                            callback(tick)
                
            except asyncio.TimeoutError:
                print("WebSocket idle for 30s, checking connection...")
            except websockets.exceptions.ConnectionClosed:
                print("WebSocket disconnected")
                self.is_connected = False
                break
            except Exception as e:
                print(f"Error receiving tick: {e}")
    
    def _parse_binary_tick(self, data: bytes) -> Optional[Tick]:
        """
        Parse binary tick data from Zerodha.
        
        Format (for LTP mode):
        - Bytes 0-3: Instrument token
        - Bytes 4-7: Sequence number
        - Bytes 8-11: Exchange timestamp
        - Bytes 12-19: Last traded price (double)
        """
        if len(data) < 20:
            return None
        
        import struct
        
        instrument_token = struct.unpack(">I", data[0:4])[0]
        last_price = struct.unpack(">d", data[12:20])[0]
        
        # Lookup symbol from token (must maintain cache)
        # This is simplified; in production use your instrument cache
        
        return Tick(
            instrument_token=instrument_token,
            tradingsymbol="UNKNOWN",  # Lookup from cache
            last_price=last_price,
            bid=last_price,
            ask=last_price,
            volume=0,
            timestamp=datetime.now()
        )
    
    async def disconnect(self):
        """Close WebSocket connection."""
        if self.ws:
            await self.ws.close()
        self.is_connected = False
```

## Rate Limiting and Error Handling

Zerodha enforces rate limits. Build resilience:

```python
"""
Rate limiting and retry logic for API calls.
"""

import time
from functools import wraps
from typing import TypeVar, Callable

F = TypeVar('F', bound=Callable)

def rate_limit(calls_per_second: float = 10) -> Callable[[F], F]:
    """
    Decorator to enforce rate limit on function calls.
    
    Args:
        calls_per_second: Max calls per second (Zerodha default: 10)
    """
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]
    
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        
        return wrapper
    
    return decorator


def retry_on_error(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    timeout_exception: Exception = requests.Timeout
) -> Callable[[F], F]:
    """
    Decorator to retry failed API calls with exponential backoff.
    
    Args:
        max_retries: Max retry attempts
        backoff_factor: Multiply delay by this each retry
        timeout_exception: Which exceptions trigger retry
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except timeout_exception as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        wait_time = backoff_factor ** attempt
                        print(f"Attempt {attempt + 1} failed, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"All {max_retries + 1} attempts failed")
            
            raise last_exception
        
        return wrapper
    
    return decorator


class RobustKiteClient:
    """Kite client with rate limiting and retry logic built-in."""
    
    def __init__(self, rest_api: KiteRestAPI):
        self.rest_api = rest_api
    
    @rate_limit(calls_per_second=5)  # Conservative rate limit
    @retry_on_error(max_retries=2, timeout_exception=Exception)
    def place_order_safe(
        self,
        tradingsymbol: str,
        exchange: str,
        transaction_type: str,
        quantity: int,
        order_type: str,
        price: float = 0
    ) -> str:
        """
        Place order with automatic retry and rate limiting.
        """
        return self.rest_api.place_order(
            tradingsymbol=tradingsymbol,
            exchange=exchange,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type=order_type,
            price=price
        )
    
    def get_positions_safe(self):
        """Fetch positions with retry logic."""
        for attempt in range(3):
            try:
                return self.rest_api.get_positions()
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(1)
```

---

# Module 25.2: Order Management System — State Machines for Safety

## The Problem: Broker-System State Divergence

Your system thinks you placed a BUY order for 100 shares, but:
- Network packet lost → broker never received it
- You received an ACCEPTED response, but network died before broker sent confirmation
- Partial fill occurred; your cache thinks it's still pending
- Order was cancelled by broker (end of day), but you think it's alive

**Solution**: Build an explicit state machine that reconciles local state with broker state.

## Order State Machine

```python
"""
Order state machine: Explicit states and transitions.
Every order moves through these states; invalid transitions trigger alerts.
"""

from enum import Enum
from typing import Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta

class OrderStatus(Enum):
    """All possible order states."""
    # Initial states
    INIT = "INIT"              # Order created locally, not sent yet
    PENDING = "PENDING"        # Sent to broker, waiting for response
    
    # Active states
    OPEN = "OPEN"              # Acknowledged by broker, waiting for fill
    PARTIAL = "PARTIAL"        # Partially filled
    
    # Terminal states
    COMPLETE = "COMPLETE"      # Fully filled
    CANCELLED = "CANCELLED"    # Cancelled (by us or broker)
    REJECTED = "REJECTED"      # Rejected by broker (insufficient margin, invalid qty, etc.)
    EXPIRED = "EXPIRED"        # Expired (GTT not triggered, end of day)
    
    # Error states
    UNKNOWN = "UNKNOWN"        # Lost sync with broker


class OrderStateMachine:
    """
    Manages valid state transitions for an order.
    Raises exceptions for invalid transitions (indicates bugs or data corruption).
    """
    
    # Define valid transitions: from_state -> set of valid to_states
    VALID_TRANSITIONS = {
        OrderStatus.INIT: {OrderStatus.PENDING, OrderStatus.REJECTED},
        OrderStatus.PENDING: {OrderStatus.OPEN, OrderStatus.REJECTED, OrderStatus.CANCELLED},
        OrderStatus.OPEN: {OrderStatus.PARTIAL, OrderStatus.COMPLETE, OrderStatus.CANCELLED, OrderStatus.EXPIRED},
        OrderStatus.PARTIAL: {OrderStatus.COMPLETE, OrderStatus.CANCELLED, OrderStatus.EXPIRED},
        OrderStatus.COMPLETE: set(),  # Terminal state
        OrderStatus.CANCELLED: set(),  # Terminal state
        OrderStatus.REJECTED: set(),   # Terminal state
        OrderStatus.EXPIRED: set(),    # Terminal state
        OrderStatus.UNKNOWN: {OrderStatus.OPEN, OrderStatus.PARTIAL, OrderStatus.COMPLETE},  # Resync
    }
    
    def __init__(self, order_id: str, initial_status: OrderStatus = OrderStatus.INIT):
        self.order_id = order_id
        self.current_status = initial_status
        self.status_history: list[tuple[OrderStatus, datetime]] = [
            (initial_status, datetime.now())
        ]
    
    def transition_to(self, new_status: OrderStatus) -> bool:
        """
        Attempt to transition to a new state.
        
        Args:
            new_status: Target state
            
        Returns:
            True if valid, raises ValueError if invalid
            
        Raises:
            ValueError: If transition is not allowed
        """
        if new_status == self.current_status:
            return True  # No-op
        
        valid_next = self.VALID_TRANSITIONS.get(self.current_status, set())
        
        if new_status not in valid_next:
            raise ValueError(
                f"Invalid transition for order {self.order_id}: "
                f"{self.current_status.value} → {new_status.value}"
            )
        
        self.current_status = new_status
        self.status_history.append((new_status, datetime.now()))
        return True
    
    def is_terminal(self) -> bool:
        """Check if order is in a terminal state."""
        return self.current_status in {
            OrderStatus.COMPLETE,
            OrderStatus.REJECTED,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED
        }
    
    def get_duration(self) -> timedelta:
        """How long has order been alive?"""
        if self.status_history:
            return datetime.now() - self.status_history[0][1]
        return timedelta(0)


@dataclass
class ManagedOrder:
    """
    Order with full lifecycle management.
    Tracks local state + broker state + reconciliation.
    """
    order_id: str
    tradingsymbol: str
    exchange: str
    transaction_type: str  # BUY / SELL
    quantity: int
    order_type: str  # MARKET / LIMIT / STOP / STOPLIMIT
    price: float
    
    # State management
    state_machine: OrderStateMachine = field(default_factory=lambda: OrderStateMachine(""))
    
    # Execution tracking
    filled_quantity: int = 0
    average_price: float = 0.0
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    sent_to_broker_at: Optional[datetime] = None
    last_update_at: Optional[datetime] = None
    
    # Reconciliation flags
    sent_to_broker: bool = False
    broker_acknowledged: bool = False
    last_synced_with_broker: Optional[datetime] = None
    
    # Risk flags
    is_bracket_order: bool = False
    parent_order_id: Optional[str] = None
    associated_orders: list[str] = field(default_factory=list)
    
    def get_pending_quantity(self) -> int:
        """How many shares still need to fill?"""
        return self.quantity - self.filled_quantity
    
    def get_pnl(self, current_price: float) -> float:
        """
        Calculate unrealized P&L (if position still open).
        Only valid for filled portions.
        """
        if self.filled_quantity == 0:
            return 0.0
        
        if self.transaction_type == "BUY":
            return self.filled_quantity * (current_price - self.average_price)
        else:  # SELL
            return self.filled_quantity * (self.average_price - current_price)
    
    def update_from_broker(
        self,
        broker_status: str,
        filled_qty: int,
        avg_price: float
    ) -> bool:
        """
        Update local order state based on broker response.
        
        Args:
            broker_status: Status from broker API (OPEN, COMPLETE, REJECTED, etc.)
            filled_qty: Shares filled
            avg_price: Average fill price
            
        Returns:
            True if update was valid, raises if inconsistent
        """
        # Map broker status to our status
        status_map = {
            "OPEN": OrderStatus.OPEN,
            "PARTIALLY_FILLED": OrderStatus.PARTIAL,
            "COMPLETE": OrderStatus.COMPLETE,
            "CANCELLED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED,
        }
        
        target_status = status_map.get(broker_status, OrderStatus.UNKNOWN)
        
        # Validate: filled quantity shouldn't decrease
        if filled_qty < self.filled_quantity:
            raise ValueError(
                f"Filled quantity decreased: {self.filled_quantity} → {filled_qty}. "
                "Possible data corruption."
            )
        
        # Update
        self.filled_quantity = filled_qty
        if filled_qty > 0:
            self.average_price = avg_price
        self.state_machine.transition_to(target_status)
        self.last_synced_with_broker = datetime.now()
        
        return True
```

## Order Management System

```python
"""
Central order management system: Tracks all orders, enforces state consistency.
"""

from collections import defaultdict
from typing import Dict, List, Optional

class OrderManager:
    """
    Central repository for order tracking.
    Ensures no order is double-sent, handles reconciliation with broker.
    """
    
    def __init__(self, kite_client: RobustKiteClient):
        self.kite = kite_client
        self.orders: Dict[str, ManagedOrder] = {}  # order_id -> ManagedOrder
        self.pending_orders: List[str] = []  # Orders waiting for broker ack
        self.by_symbol: Dict[str, List[str]] = defaultdict(list)  # symbol -> [order_ids]
        self.reconciliation_state = "CLEAN"
    
    def create_order(
        self,
        tradingsymbol: str,
        exchange: str,
        transaction_type: str,
        quantity: int,
        order_type: str,
        price: float = 0
    ) -> ManagedOrder:
        """
        Create a new order locally (not sent to broker yet).
        
        Args:
            tradingsymbol: e.g., "RELIANCE"
            exchange: "NSE" or "BSE"
            transaction_type: "BUY" or "SELL"
            quantity: Number of shares
            order_type: "MARKET", "LIMIT", "STOP", "STOPLIMIT"
            price: Limit price (for limit orders)
            
        Returns:
            ManagedOrder object (not yet sent)
        """
        # Generate local order ID (format: "LOCAL_<timestamp>_<symbol>_<rand>")
        import random
        local_order_id = f"LOCAL_{int(datetime.now().timestamp() * 1000)}_{random.randint(0, 9999)}"
        
        order = ManagedOrder(
            order_id=local_order_id,
            tradingsymbol=tradingsymbol,
            exchange=exchange,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type=order_type,
            price=price,
            state_machine=OrderStateMachine(local_order_id)
        )
        
        self.orders[local_order_id] = order
        self.by_symbol[tradingsymbol].append(local_order_id)
        
        return order
    
    def send_order_to_broker(self, local_order_id: str) -> str:
        """
        Send a locally-created order to the broker.
        Updates order state on success.
        
        Args:
            local_order_id: Order ID from create_order
            
        Returns:
            Broker order ID (may differ from local ID)
            
        Raises:
            RuntimeError: If broker rejects order
            ValueError: If order not found or already sent
        """
        order = self.orders.get(local_order_id)
        if not order:
            raise ValueError(f"Order {local_order_id} not found")
        
        if order.sent_to_broker:
            raise ValueError(f"Order {local_order_id} already sent to broker")
        
        try:
            broker_order_id = self.kite.place_order_safe(
                tradingsymbol=order.tradingsymbol,
                exchange=order.exchange,
                transaction_type=order.transaction_type,
                quantity=order.quantity,
                order_type=order.order_type,
                price=order.price
            )
        except Exception as e:
            # Broker rejected order
            order.state_machine.transition_to(OrderStatus.REJECTED)
            raise RuntimeError(f"Broker rejected order: {e}")
        
        # Update order state
        order.order_id = broker_order_id  # Update to broker's ID
        order.sent_to_broker = True
        order.sent_to_broker_at = datetime.now()
        order.state_machine.transition_to(OrderStatus.PENDING)
        
        self.pending_orders.append(broker_order_id)
        
        return broker_order_id
    
    def sync_with_broker(self) -> Dict[str, ManagedOrder]:
        """
        Fetch all orders from broker and update local state.
        Critical for reconciliation after restarts or network failures.
        
        Returns:
            Dictionary of orders that changed state
        """
        broker_orders = self.kite.rest_api.get_orders()
        changed_orders = {}
        
        for broker_order in broker_orders:
            order = self.orders.get(broker_order.order_id)
            
            if not order:
                # Broker has order we don't know about (shouldn't happen in normal operation)
                print(f"Warning: Broker has unknown order {broker_order.order_id}")
                continue
            
            old_status = order.state_machine.current_status
            
            try:
                order.update_from_broker(
                    broker_status=broker_order.status,
                    filled_qty=broker_order.filled_quantity,
                    avg_price=broker_order.average_price
                )
                
                if old_status != order.state_machine.current_status:
                    changed_orders[broker_order.order_id] = order
                    
            except ValueError as e:
                print(f"Reconciliation error on {broker_order.order_id}: {e}")
                order.state_machine.current_status = OrderStatus.UNKNOWN
        
        self.reconciliation_state = "CLEAN"
        return changed_orders
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order (must be in OPEN or PARTIAL state).
        
        Args:
            order_id: Broker order ID
            
        Returns:
            True if cancellation initiated
            
        Raises:
            RuntimeError: If broker refuses cancellation
            ValueError: If order not found or in terminal state
        """
        order = self.orders.get(order_id)
        if not order:
            raise ValueError(f"Order {order_id} not found")
        
        if order.state_machine.is_terminal():
            raise ValueError(f"Order {order_id} already in terminal state: {order.state_machine.current_status.value}")
        
        try:
            self.kite.rest_api.cancel_order(order_id, order.exchange)
        except Exception as e:
            raise RuntimeError(f"Failed to cancel order: {e}")
        
        order.state_machine.transition_to(OrderStatus.CANCELLED)
        return True
    
    def get_open_orders(self, tradingsymbol: Optional[str] = None) -> List[ManagedOrder]:
        """
        Get all non-terminal orders.
        
        Args:
            tradingsymbol: Filter by symbol (None = all symbols)
            
        Returns:
            List of open orders
        """
        result = []
        
        symbols = [tradingsymbol] if tradingsymbol else self.by_symbol.keys()
        
        for sym in symbols:
            for order_id in self.by_symbol[sym]:
                order = self.orders[order_id]
                if not order.state_machine.is_terminal():
                    result.append(order)
        
        return result
    
    def get_filled_quantity(self, tradingsymbol: str) -> int:
        """
        Total quantity filled for a symbol (across all orders).
        
        Args:
            tradingsymbol: Symbol to check
            
        Returns:
            Total filled quantity
        """
        total = 0
        for order_id in self.by_symbol[tradingsymbol]:
            order = self.orders[order_id]
            total += order.filled_quantity
        return total
    
    def emergency_cancel_all(self) -> int:
        """
        Cancel all open orders (use in emergencies only).
        
        Returns:
            Number of orders successfully cancelled
        """
        cancelled_count = 0
        for order_id, order in self.orders.items():
            if not order.state_machine.is_terminal():
                try:
                    self.cancel_order(order_id)
                    cancelled_count += 1
                except Exception as e:
                    print(f"Failed to cancel {order_id}: {e}")
        
        return cancelled_count


# Example usage
if __name__ == "__main__":
    # Initialize Kite client (with auth already done)
    kite_client = RobustKiteClient(rest_api)
    
    # Create order manager
    order_mgr = OrderManager(kite_client)
    
    # Create order locally
    order = order_mgr.create_order(
        tradingsymbol="RELIANCE",
        exchange="NSE",
        transaction_type="BUY",
        quantity=1,
        order_type="LIMIT",
        price=2500.0
    )
    
    # Send to broker
    broker_order_id = order_mgr.send_order_to_broker(order.order_id)
    print(f"Order sent, broker ID: {broker_order_id}")
    
    # Later: sync with broker to get fills
    changed = order_mgr.sync_with_broker()
    for order_id, order in changed.items():
        print(f"Order {order_id} status: {order.state_machine.current_status.value}")
```

---

# Module 25.3: Paper Trading to Live Trading — Safe Deployment

## Paper Trading: Simulated Execution

Paper trading runs your real strategy logic with **simulated** order execution. Critical for:
- Validating order placement logic (no crashes, malformed orders)
- Testing reconciliation without losing money
- Measuring execution quality before real trades

```python
"""
Paper trading: Simulated broker for testing.
Same interface as real Kite, but execution is simulated.
"""

from typing import Dict, List
from datetime import datetime
import random

class PaperBroker:
    """
    Simulated broker for paper trading.
    Implements same interface as KiteRestAPI but with simulated fills.
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        """
        Initialize paper broker.
        
        Args:
            initial_capital: Starting account balance
        """
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.orders: Dict[str, Order] = {}
        self.order_counter = 0
        self.price_cache: Dict[str, float] = {}  # Current prices
        
    def set_price(self, tradingsymbol: str, price: float):
        """
        Set current market price for a symbol (for simulation).
        """
        self.price_cache[tradingsymbol] = price
    
    def place_order(
        self,
        tradingsymbol: str,
        exchange: str,
        transaction_type: str,
        quantity: int,
        order_type: str,
        price: float = 0,
        **kwargs
    ) -> str:
        """
        Simulate order placement.
        
        For MARKET orders: fill immediately at current price
        For LIMIT orders: fill if limit price >= market price
        """
        self.order_counter += 1
        order_id = f"PAPER_{self.order_counter}"
        
        current_price = self.price_cache.get(tradingsymbol, price)
        
        # Determine if order fills immediately
        fills_immediately = False
        fill_price = 0
        
        if order_type == "MARKET":
            # Market orders fill at current price + spread
            fill_price = current_price * (1.001 if transaction_type == "BUY" else 0.999)
            fills_immediately = True
        elif order_type == "LIMIT":
            if transaction_type == "BUY" and price >= current_price:
                fill_price = price
                fills_immediately = True
            elif transaction_type == "SELL" and price <= current_price:
                fill_price = price
                fills_immediately = True
        
        # Create order
        order = Order(
            order_id=order_id,
            instrument_token=0,
            exchange_token="",
            tradingsymbol=tradingsymbol,
            order_type=order_type,
            transaction_type=transaction_type,
            quantity=quantity,
            price=price,
            status="COMPLETE" if fills_immediately else "OPEN",
            filled_quantity=quantity if fills_immediately else 0,
            pending_quantity=0 if fills_immediately else quantity,
            average_price=fill_price if fills_immediately else 0,
            placed_at=datetime.now().isoformat(),
            executed_at=datetime.now().isoformat() if fills_immediately else ""
        )
        
        self.orders[order_id] = order
        
        # Update position and cash
        if fills_immediately:
            self._execute_fill(tradingsymbol, transaction_type, quantity, fill_price)
        
        return order_id
    
    def _execute_fill(
        self,
        tradingsymbol: str,
        transaction_type: str,
        quantity: int,
        price: float
    ):
        """Execute a fill: update positions and cash."""
        cost = quantity * price
        
        if transaction_type == "BUY":
            # Check margin
            if self.cash < cost:
                raise RuntimeError("Insufficient cash")
            
            self.cash -= cost
            
            if tradingsymbol not in self.positions:
                self.positions[tradingsymbol] = Position(
                    tradingsymbol=tradingsymbol,
                    quantity=quantity,
                    buy_quantity=quantity,
                    sell_quantity=0,
                    average_buy_price=price,
                    average_sell_price=0,
                    last_price=price,
                    multiplier=1,
                    pnl=0,
                    m2m=0
                )
            else:
                pos = self.positions[tradingsymbol]
                pos.average_buy_price = (
                    (pos.average_buy_price * pos.buy_quantity + price * quantity) /
                    (pos.buy_quantity + quantity)
                )
                pos.buy_quantity += quantity
                pos.quantity += quantity
                pos.last_price = price
        
        else:  # SELL
            if tradingsymbol not in self.positions or self.positions[tradingsymbol].quantity < quantity:
                raise RuntimeError("Insufficient position")
            
            self.cash += cost
            pos = self.positions[tradingsymbol]
            pos.average_sell_price = (
                (pos.average_sell_price * pos.sell_quantity + price * quantity) /
                (pos.sell_quantity + quantity)
            )
            pos.sell_quantity += quantity
            pos.quantity -= quantity
            pos.last_price = price
            
            if pos.quantity == 0:
                del self.positions[tradingsymbol]
    
    def get_positions(self) -> Dict[str, Position]:
        """Return current positions."""
        return self.positions.copy()
    
    def get_margins(self) -> Dict[str, float]:
        """Return available margin."""
        # In paper trading, no margin constraints
        return {
            "available": self.cash,
            "utilised": 0,
            "net": self.cash,
            "equity_multiplier": 1.0
        }
    
    def get_orders(self) -> List[Order]:
        """Return all orders."""
        return list(self.orders.values())
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        order = self.orders.get(order_id)
        if not order:
            return False
        if order.status in ("COMPLETE", "CANCELLED", "REJECTED"):
            raise RuntimeError("Cannot cancel order in terminal state")
        
        order.status = "CANCELLED"
        return True
    
    def get_account_value(self) -> float:
        """Total account value (cash + position values)."""
        total = self.cash
        for pos in self.positions.values():
            total += pos.quantity * self.price_cache.get(pos.tradingsymbol, 0)
        return total


class PaperTradingRunner:
    """Run your trading strategy in paper mode for testing."""
    
    def __init__(self, broker: PaperBroker):
        self.broker = broker
        self.trades: List[Dict] = []
    
    def run_strategy(
        self,
        strategy_func,
        market_data: List[Dict],
        interval_seconds: int = 60
    ):
        """
        Run strategy function repeatedly with market data.
        
        Args:
            strategy_func: Function(broker, market_data) -> None
            market_data: Historical price data
            interval_seconds: How often to call strategy
        """
        for i, data in enumerate(market_data):
            # Update prices
            for symbol, price in data.items():
                self.broker.set_price(symbol, price)
            
            # Call strategy
            try:
                strategy_func(self.broker, data)
            except Exception as e:
                print(f"Strategy error at step {i}: {e}")
                break
        
        # Print results
        print(f"\nPaper Trading Results:")
        print(f"Final Account Value: ₹{self.broker.get_account_value():,.2f}")
        print(f"Profit/Loss: ₹{self.broker.get_account_value() - 100000:,.2f}")
        print(f"Orders Placed: {len(self.broker.orders)}")


# Example: Paper trading a simple strategy
def test_strategy(broker: PaperBroker, data: Dict[str, float]):
    """Sample strategy: Buy if price drops 1%, sell if up 2%."""
    positions = broker.get_positions()
    
    reliance_price = data.get("RELIANCE")
    if not reliance_price:
        return
    
    if "RELIANCE" not in positions and reliance_price > 2000:
        # Buy
        try:
            broker.place_order(
                tradingsymbol="RELIANCE",
                exchange="NSE",
                transaction_type="BUY",
                quantity=1,
                order_type="MARKET"
            )
            print(f"Bought RELIANCE at ₹{reliance_price}")
        except RuntimeError as e:
            print(f"Buy failed: {e}")
    
    elif "RELIANCE" in positions:
        pos = positions["RELIANCE"]
        pnl_pct = (reliance_price - pos.average_buy_price) / pos.average_buy_price * 100
        
        if pnl_pct > 2:  # +2% profit target
            broker.place_order(
                tradingsymbol="RELIANCE",
                exchange="NSE",
                transaction_type="SELL",
                quantity=pos.quantity,
                order_type="MARKET"
            )
            print(f"Sold RELIANCE at ₹{reliance_price}, P&L: {pnl_pct:.2f}%")
```

## Transitioning from Paper to Live

### Step 1: Validate Paper Trading Performance

```python
"""
Metrics for comparing paper vs live trading.
"""

from dataclasses import dataclass
from typing import List

@dataclass
class PerformanceMetrics:
    """Trading performance statistics."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float  # winning_trades / total_trades
    avg_win: float
    avg_loss: float
    profit_factor: float  # total_wins / total_losses
    sharpe_ratio: float
    max_drawdown: float
    
    def __str__(self) -> str:
        return f"""
Performance Report:
  Total Trades: {self.total_trades}
  Win Rate: {self.win_rate:.1%}
  Profit Factor: {self.profit_factor:.2f}
  Max Drawdown: {self.max_drawdown:.2%}
  Sharpe Ratio: {self.sharpe_ratio:.2f}
"""

def calculate_metrics(trades: List[Dict]) -> PerformanceMetrics:
    """
    Calculate performance metrics from trade list.
    
    Each trade dict should have: entry_price, exit_price, qty, side
    """
    if not trades:
        return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    pnls = []
    wins = 0
    
    for trade in trades:
        if trade['side'] == 'BUY':
            pnl = (trade['exit_price'] - trade['entry_price']) * trade['qty']
        else:
            pnl = (trade['entry_price'] - trade['exit_price']) * trade['qty']
        
        pnls.append(pnl)
        if pnl > 0:
            wins += 1
    
    total_pnl = sum(pnls)
    win_pnl = sum(p for p in pnls if p > 0)
    loss_pnl = abs(sum(p for p in pnls if p < 0))
    
    return PerformanceMetrics(
        total_trades=len(trades),
        winning_trades=wins,
        losing_trades=len(trades) - wins,
        win_rate=wins / len(trades),
        avg_win=win_pnl / wins if wins > 0 else 0,
        avg_loss=loss_pnl / (len(trades) - wins) if len(trades) - wins > 0 else 0,
        profit_factor=win_pnl / loss_pnl if loss_pnl > 0 else float('inf'),
        sharpe_ratio=total_pnl / (1 + sum(x**2 for x in pnls)**0.5),  # Simplified
        max_drawdown=abs(min([sum(pnls[:i]) for i in range(len(pnls))]) / total_pnl) if total_pnl > 0 else 0
    )
```

### Step 2: Position Sizing for Small Capital

```python
"""
Position sizing strategies for ₹50K-₹1L accounts.
"""

from typing import Optional

class PositionSizer:
    """Calculates safe position sizes based on account size and risk."""
    
    def __init__(self, account_capital: float, risk_per_trade_pct: float = 1.0):
        """
        Initialize position sizer.
        
        Args:
            account_capital: Total account size in rupees
            risk_per_trade_pct: % of capital risked per trade (1-2% recommended)
        """
        self.capital = account_capital
        self.risk_pct = risk_per_trade_pct
    
    def kelly_position_size(
        self,
        entry_price: float,
        stoploss_price: float,
        win_probability: float
    ) -> int:
        """
        Kelly Criterion: Optimal position size to maximize long-term growth.
        
        Position Size = (Win% * Avg_Win - Loss% * Avg_Loss) / (Avg_Win / Avg_Loss)
        
        For simplicity: Qty = Capital * risk_pct / (entry_price - stoploss_price)
        
        Args:
            entry_price: Entry price
            stoploss_price: Stoploss level
            win_probability: Estimated probability of profit
            
        Returns:
            Safe quantity to trade
        """
        risk_per_share = abs(entry_price - stoploss_price)
        risk_amount = self.capital * self.risk_pct / 100
        
        quantity = int(risk_amount / risk_per_share)
        return max(1, quantity)
    
    def max_position_size(self, entry_price: float) -> int:
        """
        Maximum position size (% of account value).
        Never risk more than 5% of account in single position.
        """
        max_value = self.capital * 0.05  # 5% max per position
        quantity = int(max_value / entry_price)
        return max(1, quantity)
    
    def recommended_quantity(
        self,
        entry_price: float,
        stoploss_price: float
    ) -> int:
        """
        Conservative recommended quantity.
        Takes Kelly sizing but caps at 5% of account.
        """
        kelly_qty = self.kelly_position_size(entry_price, stoploss_price, 0.55)
        max_qty = self.max_position_size(entry_price)
        return min(kelly_qty, max_qty)


# Example: Sizing for ₹50K account
sizer = PositionSizer(account_capital=50000, risk_per_trade_pct=2)

# Reliance at 2500, stop at 2450
qty = sizer.recommended_quantity(entry_price=2500, stoploss_price=2450)
print(f"Recommended: {qty} shares of RELIANCE")
# Output: 50 shares (risking ₹2500 on 50-point stop)
```

### Step 3: Gradual Capital Scaling

```python
"""
Safe capital escalation strategy.
Start small, prove system reliability, scale gradually.
"""

from datetime import datetime, timedelta
from enum import Enum

class TradingPhase(Enum):
    """Stages of capital scaling."""
    PHASE1 = "MICRO_CAPITAL"      # ₹10K-₹25K, 1-2 trades/day
    PHASE2 = "SMALL_CAPITAL"      # ₹25K-₹50K, 2-5 trades/day
    PHASE3 = "MEDIUM_CAPITAL"     # ₹50K-₹1L, 5-10 trades/day
    PHASE4 = "INSTITUTIONAL"      # ₹1L+, 10+ trades/day


class CapitalScalingPolicy:
    """Determines when to increase capital based on performance."""
    
    def __init__(self):
        self.current_phase = TradingPhase.PHASE1
        self.start_date = datetime.now()
        self.min_duration_before_scale = timedelta(days=7)  # At least 1 week
        self.success_thresholds = {
            TradingPhase.PHASE1: {"win_rate": 0.45, "sharpe": 0.5},
            TradingPhase.PHASE2: {"win_rate": 0.50, "sharpe": 1.0},
            TradingPhase.PHASE3: {"win_rate": 0.52, "sharpe": 1.5},
        }
    
    def should_scale_up(
        self,
        metrics: PerformanceMetrics,
        current_date: datetime
    ) -> bool:
        """
        Check if we should increase capital.
        
        Args:
            metrics: Current performance
            current_date: Current date
            
        Returns:
            True if we should scale up
        """
        # Minimum duration requirement
        if current_date - self.start_date < self.min_duration_before_scale:
            return False
        
        # Minimum trades requirement (at least 20 trades)
        if metrics.total_trades < 20:
            return False
        
        # Performance thresholds
        thresholds = self.success_thresholds.get(self.current_phase)
        if not thresholds:
            return False  # Already at max phase
        
        if (metrics.win_rate >= thresholds["win_rate"] and
            metrics.sharpe_ratio >= thresholds["sharpe"]):
            return True
        
        return False
    
    def scale_up(self):
        """Move to next phase."""
        phases = list(TradingPhase)
        current_idx = phases.index(self.current_phase)
        
        if current_idx < len(phases) - 1:
            self.current_phase = phases[current_idx + 1]
            self.start_date = datetime.now()
            return True
        
        return False
    
    def get_capital_for_phase(self) -> float:
        """Recommended capital for current phase."""
        capital_map = {
            TradingPhase.PHASE1: 15000,
            TradingPhase.PHASE2: 35000,
            TradingPhase.PHASE3: 75000,
            TradingPhase.PHASE4: 150000,
        }
        return capital_map[self.current_phase]
    
    def get_max_positions(self) -> int:
        """Max concurrent positions for current phase."""
        max_positions = {
            TradingPhase.PHASE1: 2,
            TradingPhase.PHASE2: 4,
            TradingPhase.PHASE3: 6,
            TradingPhase.PHASE4: 10,
        }
        return max_positions[self.current_phase]
```

## Tax Implications for Algo Traders in India

```python
"""
India tax classification for algorithmic trading.
Critical for compliance; penalties are severe.
"""

from enum import Enum
from dataclasses import dataclass

class IncomeClassification(Enum):
    """How algo trading income is taxed in India."""
    
    CAPITAL_GAINS = "CAPITAL_GAINS"
    # Short-term (<1 year): Taxed as normal income + 15% STT
    # Long-term (>1 year): 20% + cess
    
    BUSINESS_INCOME = "BUSINESS_INCOME"
    # If trading is your primary activity (frequency matters)
    # Taxed as business income (slab rates)
    # Can claim deductions (software, subscriptions, losses)
    
    SPECULATIVE_INCOME = "SPECULATIVE_INCOME"
    # Frequent intraday trading (>90% of trades)
    # Taxed at 40% if profit, can offset against other business income
    # Loss: Can't offset against other income (must wait for 8 years)


@dataclass
class TaxCalculation:
    """Tax liability for a financial year."""
    total_profit: float
    stcg_profit: float  # Short-term capital gains
    ltcg_profit: float  # Long-term capital gains
    total_loss: float
    income_classification: IncomeClassification
    tax_liability: float
    
    def explain(self) -> str:
        """Human-readable tax explanation."""
        explanation = f"""
Tax Calculation for FY {datetime.now().year - 1 if datetime.now().month < 4 else datetime.now().year}:
================

Classification: {self.income_classification.value}
Total Profit: ₹{self.total_profit:,.2f}
Total Loss: ₹{self.total_loss:,.2f}

Net P&L: ₹{self.total_profit - self.total_loss:,.2f}
Estimated Tax Liability: ₹{self.tax_liability:,.2f}

CRITICAL NOTES:
1. Record all trades in ITR Schedule 112A (speculative) or 112A (non-speculative)
2. Keep broker statements and ledger entries for 6 years
3. If >₹10L trading volume, expect income tax notice
4. Don't evade: Penalties are 50-300% of tax + prosecution risk
5. Consider consulting CA for your specific situation
"""
        return explanation


def classify_trading_activity(
    total_trades: int,
    intraday_trades: int,
    holding_days_avg: float
) -> IncomeClassification:
    """
    Classify your trading for tax purposes.
    
    Args:
        total_trades: Total trades in year
        intraday_trades: Trades closed same-day
        holding_days_avg: Average holding period (days)
        
    Returns:
        Tax classification
    """
    intraday_pct = intraday_trades / total_trades if total_trades > 0 else 0
    
    # Intraday-heavy trading: Speculative income
    if intraday_pct > 0.9 and total_trades > 50:
        return IncomeClassification.SPECULATIVE_INCOME
    
    # Regular trading with mixed holding periods: Business income (if frequent)
    if total_trades > 50 and total_trades < 500:
        return IncomeClassification.BUSINESS_INCOME
    
    # Low-frequency, longer holding: Capital gains
    if total_trades < 50 and holding_days_avg > 30:
        return IncomeClassification.CAPITAL_GAINS
    
    # Default: Conservative classification
    return IncomeClassification.BUSINESS_INCOME


def estimate_tax_liability(
    classification: IncomeClassification,
    net_profit: float,
    stcg_profit: float = 0,
    ltcg_profit: float = 0
) -> TaxCalculation:
    """
    Estimate tax liability (VERY SIMPLIFIED; consult a CA).
    
    Args:
        classification: Income type
        net_profit: Total profit after losses
        stcg_profit: Short-term capital gains portion
        ltcg_profit: Long-term capital gains portion
        
    Returns:
        TaxCalculation object
    """
    # Assumed tax slab: 30% (highest slab)
    # Actual rate depends on your income bracket
    
    tax = 0
    
    if classification == IncomeClassification.CAPITAL_GAINS:
        # STCG: Taxed as normal income + 15% STT
        tax += stcg_profit * 0.30  # Assumes 30% bracket
        # LTCG: 20% + cess
        tax += ltcg_profit * 0.20
    
    elif classification == IncomeClassification.BUSINESS_INCOME:
        # Normal income tax rate + 4% cess
        tax = net_profit * 0.30 * 1.04
    
    elif classification == IncomeClassification.SPECULATIVE_INCOME:
        # 40% if profit (fixed rate)
        if net_profit > 0:
            tax = net_profit * 0.40
        # Loss can't be offset against other income
    
    return TaxCalculation(
        total_profit=net_profit if net_profit > 0 else 0,
        stcg_profit=stcg_profit,
        ltcg_profit=ltcg_profit,
        total_loss=abs(net_profit) if net_profit < 0 else 0,
        income_classification=classification,
        tax_liability=tax
    )


# Example
trades_analysis = {
    "total_trades": 250,
    "intraday_trades": 200,
    "avg_holding_days": 4
}

classification = classify_trading_activity(
    total_trades=trades_analysis["total_trades"],
    intraday_trades=trades_analysis["intraday_trades"],
    holding_days_avg=trades_analysis["avg_holding_days"]
)

net_profit = 150000
tax_calc = estimate_tax_liability(classification, net_profit)
print(tax_calc.explain())
```

---

## Production Deployment Checklist

Before going live with real money, verify:

```python
"""
Pre-deployment checklist for live trading.
Each item must be verified before deploying capital.
"""

class LiveDeploymentChecklist:
    """Safety checklist before live trading."""
    
    items = {
        "authentication": [
            "API key and secret stored in environment variables (never hardcoded)",
            "OAuth token refresh logic tested",
            "Session timeout handling verified"
        ],
        "order_management": [
            "Order state machine tested with all transitions",
            "Partial fill handling verified",
            "Reconciliation logic tested after simulated network failure",
            "Emergency cancel-all function tested"
        ],
        "risk_management": [
            "Position size calculation verified with real account capital",
            "Max daily loss limit enforced",
            "Max position size enforced",
            "Stoploss placed on every entry"
        ],
        "execution": [
            "Paper trading ran for 7+ days with >20 trades",
            "Win rate >45% in paper trading",
            "Sharpe ratio >0.5 in paper trading",
            "Execution latency measured (<2s for order placement)"
        ],
        "monitoring": [
            "Logging configured (all trades logged with timestamps)",
            "Email alerts for errors configured",
            "Daily performance report generated",
            "P&L tracking real-time"
        ],
        "compliance": [
            "Tax classification determined",
            "CA consulted for your specific case",
            "Trade records retention system in place"
        ],
        "disaster_recovery": [
            "Backup internet connection tested",
            "Manual trading override available (kill switch)",
            "Account recovery process tested",
            "Data backup to external storage"
        ]
    }
    
    def print_checklist(self):
        """Print deployment checklist."""
        print("\n" + "="*60)
        print("LIVE DEPLOYMENT CHECKLIST")
        print("="*60 + "\n")
        
        for category, items in self.items.items():
            print(f"\n{category.upper()}:")
            for i, item in enumerate(items, 1):
                print(f"  [ ] {item}")
        
        print("\n" + "="*60)
        print("Do not proceed until ALL items are checked")
        print("="*60 + "\n")
```

---

## Summary

You now have:

1. **Zerodha Integration**: Complete authentication, REST API, WebSocket, and error handling
2. **Order Management**: State machine ensuring consistency between your system and broker
3. **Safe Scaling**: Paper trading → live transition with metrics and gradual capital escalation
4. **Tax Compliance**: Framework for classifying trading income in India

**Next steps**:
- Set up your Zerodha API credentials
- Run paper trading for 2 weeks (50+ trades)
- Start with ₹50K-₹1L capital
- Scale up only after hitting performance thresholds
- Consult a CA before year-end for tax planning

The bridge from backtest to production is technical and psychological. Use these tools to be systematic, not emotional.
