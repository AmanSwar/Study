# Chapter 24: Trading System Architecture

## Introduction

Building a quantitative trading system is not merely about developing sophisticated machine learning models—it's fundamentally an **engineering problem**. A state-of-the-art prediction model with 65% accuracy is worthless if the system crashes during market hours, loses trades to latency, or executes orders incorrectly due to system failures. This chapter bridges the gap between model development and production trading by teaching you to architect a robust, fault-tolerant, and observable trading system.

We assume you have strong ML/deep learning expertise and systems engineering knowledge but zero finance domain knowledge. We're building specifically for the NSE (National Stock Exchange of India) using Zerodha, India's largest retail brokerage.

### Chapter Learning Objectives

By the end of this chapter, you will:

1. **Understand system architecture principles** for live trading: event-driven architecture, separation of concerns, and fault tolerance
2. **Design and implement real-time data infrastructure** that handles WebSocket feeds, reconnections, and feature computation
3. **Build comprehensive logging and monitoring systems** with dashboards, alerts, and correlation IDs for debugging production issues
4. **Deploy production-grade code** with type hints, proper error handling, and observability patterns

---

# Module 24.1: System Design Principles and Architecture

## 24.1.1 The Complete Trading System: Component Architecture

A production trading system consists of five core components working in concert:

```
                    ┌─────────────────────────────────────────────┐
                    │        DATA INGESTION LAYER                  │
                    │   (Zerodha WebSocket, Historical Data)       │
                    └──────────────┬──────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
        ┌───────────▼────────────────┐  ┌────────▼──────────────────┐
        │   SIGNAL GENERATION LAYER   │  │  FEATURE STORE / CACHE    │
        │   (ML Models, Indicators)   │  │  (Real-time Features)     │
        └───────────┬────────────────┘  └────────┬──────────────────┘
                    │                             │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  PORTFOLIO & RISK LAYER      │
                    │  (Position Management, Risk) │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   EXECUTION LAYER           │
                    │   (Order Management,        │
                    │    Zerodha API, Fills)     │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   MONITORING LAYER          │
                    │   (Logs, Metrics, Alerts)   │
                    └─────────────────────────────┘
```

Each component is:
- **Asynchronous**: Uses async I/O to avoid blocking
- **Independently testable**: Can be unit tested in isolation
- **Fault-tolerant**: Graceful degradation when dependencies fail
- **Observable**: Logs, metrics, and alerts for production visibility

## 24.1.2 Event-Driven Architecture for Real-Time Systems

Traditional request-response patterns create latency bottlenecks. Real-time trading demands **event-driven architecture** where components react to events (price ticks, signals, order fills) rather than polling.

### Event Types and Flow

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
from enum import Enum
import uuid

class EventType(Enum):
    """Core event types in trading system"""
    MARKET_TICK = "market_tick"
    SIGNAL_GENERATED = "signal_generated"
    POSITION_UPDATED = "position_updated"
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_REJECTED = "order_rejected"
    RISK_VIOLATION = "risk_violation"
    SYSTEM_ALERT = "system_alert"

@dataclass
class TradingEvent:
    """Base event class with tracing metadata"""
    event_type: EventType
    timestamp: datetime
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure timestamps are UTC"""
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=None)

@dataclass
class MarketTickEvent(TradingEvent):
    """Market tick event from Zerodha"""
    def __init__(self, symbol: str, ltp: float, bid: float, ask: float,
                 volume: int, timestamp: datetime, **kwargs):
        super().__init__(
            event_type=EventType.MARKET_TICK,
            timestamp=timestamp,
            data={
                'symbol': symbol,
                'ltp': ltp,  # Last traded price
                'bid': bid,
                'ask': ask,
                'volume': volume,
                **kwargs
            }
        )

@dataclass
class SignalEvent(TradingEvent):
    """Signal generated by ML model"""
    def __init__(self, symbol: str, signal: int, confidence: float,
                 model_name: str, timestamp: datetime, **kwargs):
        # signal: 1 (buy), -1 (sell), 0 (no trade)
        super().__init__(
            event_type=EventType.SIGNAL_GENERATED,
            timestamp=timestamp,
            data={
                'symbol': symbol,
                'signal': signal,
                'confidence': confidence,
                'model_name': model_name,
                **kwargs
            }
        )

@dataclass
class OrderFilledEvent(TradingEvent):
    """Notification that order was filled"""
    def __init__(self, order_id: str, symbol: str, quantity: int,
                 fill_price: float, timestamp: datetime, **kwargs):
        super().__init__(
            event_type=EventType.ORDER_FILLED,
            timestamp=timestamp,
            data={
                'order_id': order_id,
                'symbol': symbol,
                'quantity': quantity,
                'fill_price': fill_price,
                **kwargs
            }
        )

# Event bus using asyncio
import asyncio
from typing import Callable, List

class EventBus:
    """Asynchronous event bus for inter-component communication"""
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._event_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
    
    def subscribe(self, event_type: EventType, handler: Callable):
        """Register handler for event type"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
    
    async def publish(self, event: TradingEvent):
        """Publish event to all subscribers"""
        await self._event_queue.put(event)
    
    async def start(self):
        """Start processing events"""
        while True:
            event = await self._event_queue.get()
            handlers = self._subscribers.get(event.event_type, [])
            
            # Run handlers concurrently
            await asyncio.gather(
                *[handler(event) for handler in handlers],
                return_exceptions=True
            )
```

## 24.1.3 Separation of Concerns: Independent Processes

Each component should be independently deployable and testable:

```python
from abc import ABC, abstractmethod
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class SignalComponent(ABC):
    """Signal generation component (independent process)"""
    
    @abstractmethod
    async def on_tick(self, event: MarketTickEvent) -> Optional[SignalEvent]:
        """Generate signal on market tick"""
        pass

class RiskComponent(ABC):
    """Risk management component (independent process)"""
    
    @abstractmethod
    async def validate_order(self, order: Dict[str, Any]) -> bool:
        """Validate order against risk limits"""
        pass
    
    @abstractmethod
    async def check_position_limits(self, symbol: str, quantity: int) -> bool:
        """Ensure position doesn't exceed limit"""
        pass

class ExecutionComponent(ABC):
    """Order execution component (independent process)"""
    
    @abstractmethod
    async def place_order(self, order: Dict[str, Any]) -> str:
        """Place order, return order_id"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order by ID"""
        pass

class PortfolioComponent(ABC):
    """Portfolio tracking component (independent process)"""
    
    @abstractmethod
    async def update_position(self, event: OrderFilledEvent):
        """Update position on fill"""
        pass
    
    @abstractmethod
    async def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get current positions"""
        pass

# Example: Signal Component Implementation
class MLSignalComponent(SignalComponent):
    """ML-based signal generation"""
    
    def __init__(self, model, event_bus: EventBus, logger: logging.Logger):
        self.model = model
        self.event_bus = event_bus
        self.logger = logger
    
    async def on_tick(self, event: MarketTickEvent) -> Optional[SignalEvent]:
        try:
            symbol = event.data['symbol']
            price = event.data['ltp']
            
            # Compute features (from cache or real-time)
            features = await self._compute_features(symbol, price)
            
            # Generate signal
            signal, confidence = self.model.predict(features)
            
            if confidence > 0.55:  # Confidence threshold
                signal_event = SignalEvent(
                    symbol=symbol,
                    signal=signal,
                    confidence=confidence,
                    model_name=self.model.name,
                    timestamp=event.timestamp
                )
                
                # Publish for downstream components
                await self.event_bus.publish(signal_event)
                return signal_event
            
        except Exception as e:
            self.logger.error(f"Signal generation failed: {e}",
                            extra={'correlation_id': event.correlation_id})
        
        return None
    
    async def _compute_features(self, symbol: str, price: float):
        """Compute features (placeholder)"""
        # In production, fetch from feature store cache
        return {}
```

## 24.1.4 Fault Tolerance: Handling Component Failures

What happens when each component fails?

### Failure Scenarios and Recovery

```python
from enum import Enum
from typing import Awaitable
import time

class ComponentHealth(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"

class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half-open
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        self.failure_count = 0
        self.state = "closed"
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"

# Fault Tolerance Strategies

class FaultToleranceStrategy:
    """Strategies for handling component failures"""
    
    # 1. Data Feed Failure
    # Strategy: Fall back to cached data, alert operator, stop trading
    async def handle_data_feed_failure(self, duration: int):
        """Handle loss of market data feed"""
        logger.critical(f"Data feed lost for {duration}s, entering safe mode")
        # Actions:
        # - Stop sending new signals
        # - Maintain stale data for risk checks only
        # - Alert trading desk
        # - Check reconnection status every 5 seconds
    
    # 2. Signal Generation Failure
    # Strategy: No signal = no trade (safe default)
    async def handle_signal_failure(self):
        """Handle signal generation timeout"""
        logger.error("Signal generation failed, treating as NO_SIGNAL")
        # Actions:
        # - Don't place orders
        # - Alert monitoring system
        # - Increment error counter
    
    # 3. Order Execution Failure
    # Strategy: Retry with exponential backoff, abort if persistent
    async def handle_execution_failure(self, order_id: str, attempt: int = 0):
        """Retry failed order placement"""
        max_attempts = 3
        backoff = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
        
        if attempt >= max_attempts:
            logger.critical(f"Order {order_id} failed after {max_attempts} attempts")
            # Actions:
            # - Escalate to human trader
            # - Log for post-trade analysis
            return False
        
        logger.warning(f"Retrying order {order_id}, attempt {attempt + 1}")
        await asyncio.sleep(backoff)
        return await self.place_order_retry(order_id, attempt + 1)
    
    # 4. Position Tracking Failure
    # Strategy: Rebuild from order history
    async def rebuild_positions_from_history(self):
        """Reconstruct positions from order/fill history"""
        logger.info("Rebuilding positions from order history")
        # Actions:
        # - Query all filled orders from database
        # - Recalculate position per symbol
        # - Compare with broker state
        # - Alert if mismatch detected
```

## 24.1.5 Database Design for Live Trading

Live trading requires a schema optimized for writes, reads, and historical analysis:

```python
from sqlalchemy import (
    create_engine, Column, String, Integer, Float, DateTime,
    ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import pytz

Base = declarative_base()

class Order(Base):
    """Orders placed by the system"""
    __tablename__ = 'orders'
    
    order_id = Column(String(50), primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    side = Column(String(10), nullable=False)  # BUY, SELL
    status = Column(String(20), nullable=False, index=True)  # PENDING, FILLED, REJECTED, CANCELLED
    order_type = Column(String(20))  # MARKET, LIMIT
    correlation_id = Column(String(50), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Indexes for fast queries
    __table_args__ = (
        Index('idx_symbol_created', 'symbol', 'created_at'),
        Index('idx_status_created', 'status', 'created_at'),
        Index('idx_correlation_id', 'correlation_id'),
    )

class Fill(Base):
    """Executed fills (partial or complete)"""
    __tablename__ = 'fills'
    
    fill_id = Column(String(50), primary_key=True)
    order_id = Column(String(50), ForeignKey('orders.order_id'), index=True)
    symbol = Column(String(20), nullable=False, index=True)
    quantity = Column(Integer, nullable=False)
    fill_price = Column(Float, nullable=False)
    fill_commission = Column(Float, default=0)
    filled_at = Column(DateTime, nullable=False, index=True)
    exchange_timestamp = Column(DateTime, nullable=False)  # Broker's timestamp
    
    __table_args__ = (
        Index('idx_symbol_filled_at', 'symbol', 'filled_at'),
    )

class Position(Base):
    """Current positions (refreshed real-time)"""
    __tablename__ = 'positions'
    
    symbol = Column(String(20), primary_key=True)
    quantity = Column(Integer, nullable=False)
    average_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=False)
    marked_to_market = Column(Float)  # Current value
    unrealized_pnl = Column(Float)
    realized_pnl = Column(Float, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class DailyPnL(Base):
    """Daily P&L snapshot"""
    __tablename__ = 'daily_pnl'
    
    date = Column(String(10), primary_key=True)  # YYYY-MM-DD
    realized_pnl = Column(Float, default=0)
    unrealized_pnl = Column(Float, default=0)
    total_pnl = Column(Float, default=0)
    snapshot_time = Column(DateTime, default=datetime.utcnow)

class RiskMetric(Base):
    """Real-time risk metrics"""
    __tablename__ = 'risk_metrics'
    
    metric_id = Column(String(50), primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    metric_name = Column(String(50), nullable=False)  # max_position, daily_loss, leverage
    metric_value = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    violated = Column(Integer, default=0)  # Boolean: 1 if violated
    
    __table_args__ = (
        Index('idx_metric_timestamp', 'metric_name', 'timestamp'),
    )

# Database connection with connection pooling
class DatabaseManager:
    """Manages database connections and transactions"""
    
    def __init__(self, database_url: str):
        # Use connection pooling for concurrent access
        self.engine = create_engine(
            database_url,
            pool_size=20,  # Max 20 concurrent connections
            max_overflow=10,  # Additional overflow connections
            pool_pre_ping=True  # Test connections before using
        )
        self.SessionLocal = sessionmaker(bind=self.engine)
        self._create_tables()
    
    def _create_tables(self):
        """Create all tables"""
        Base.metadata.create_all(self.engine)
    
    async def insert_order(self, order: Dict[str, Any]) -> str:
        """Insert order with transaction"""
        session = self.SessionLocal()
        try:
            db_order = Order(**order)
            session.add(db_order)
            session.commit()
            return db_order.order_id
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()
    
    async def insert_fill(self, fill: Dict[str, Any]):
        """Insert fill and update position"""
        session = self.SessionLocal()
        try:
            db_fill = Fill(**fill)
            session.add(db_fill)
            
            # Atomically update position
            symbol = fill['symbol']
            quantity = fill['quantity']
            fill_price = fill['fill_price']
            
            position = session.query(Position).filter_by(symbol=symbol).first()
            if position:
                # Update average price
                new_quantity = position.quantity + quantity
                position.average_price = (
                    (position.average_price * position.quantity + 
                     fill_price * quantity) / new_quantity
                )
                position.quantity = new_quantity
            else:
                position = Position(
                    symbol=symbol,
                    quantity=quantity,
                    average_price=fill_price,
                    current_price=fill_price
                )
                session.add(position)
            
            session.commit()
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()
    
    async def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all positions"""
        session = self.SessionLocal()
        try:
            positions = session.query(Position).all()
            return {
                p.symbol: {
                    'quantity': p.quantity,
                    'avg_price': p.average_price,
                    'current_price': p.current_price,
                    'unrealized_pnl': p.unrealized_pnl
                }
                for p in positions
            }
        finally:
            session.close()

```

---

# Module 24.2: Data Infrastructure for Live Trading

## 24.2.1 Real-Time Data Ingestion from Zerodha WebSocket

Zerodha provides real-time market data via WebSocket. The KiteConnect API uses a binary protocol for low-latency streaming.

### WebSocket Connection Management

```python
import websockets
import struct
import json
from typing import Callable, Optional, Dict, Any
import asyncio
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ZerodhaDataFeed:
    """Zerodha WebSocket data feed manager"""
    
    ZERODHA_SOCKET_URL = "wss://ws.kite.trade"
    HEARTBEAT_INTERVAL = 3  # seconds
    RECONNECT_BACKOFF = [1, 2, 5, 10, 30]  # Exponential backoff
    
    def __init__(self, api_key: str, access_token: str):
        self.api_key = api_key
        self.access_token = access_token
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.subscribed_instruments = set()
        self.tick_callbacks: Dict[str, Callable] = {}
        self.reconnect_attempt = 0
        self.last_heartbeat = datetime.utcnow()
        self.data_feed_healthy = True
    
    async def connect(self):
        """Establish WebSocket connection with auth"""
        try:
            self.websocket = await websockets.connect(
                self.ZERODHA_SOCKET_URL,
                ping_interval=20,  # Ping every 20s
                ping_timeout=10,
                max_size=2**20  # 1MB max frame size
            )
            
            # Send auth message
            await self._send_auth()
            
            # Start receiving data
            asyncio.create_task(self._receive_loop())
            
            # Start heartbeat monitor
            asyncio.create_task(self._heartbeat_monitor())
            
            self.reconnect_attempt = 0
            self.data_feed_healthy = True
            logger.info("Zerodha WebSocket connected and authenticated")
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            await self._handle_reconnection()
    
    async def _send_auth(self):
        """Send authentication message"""
        auth_msg = {
            "a": "authorize",
            "token": self.access_token,
            "user_id": self.api_key
        }
        await self.websocket.send(json.dumps(auth_msg))
        
        # Receive auth response
        response = await self.websocket.recv()
        auth_response = json.loads(response)
        
        if auth_response.get("type") != "connection":
            raise Exception("Authentication failed")
    
    async def subscribe(self, instrument_token: int, on_tick: Callable):
        """Subscribe to instrument updates"""
        self.subscribed_instruments.add(instrument_token)
        self.tick_callbacks[instrument_token] = on_tick
        
        # Send subscription message
        subscribe_msg = {
            "a": "subscribe",
            "v": list(self.subscribed_instruments)
        }
        
        if self.websocket:
            await self.websocket.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to instrument {instrument_token}")
    
    async def _receive_loop(self):
        """Main receive loop for market data"""
        while True:
            try:
                if not self.websocket:
                    await asyncio.sleep(1)
                    continue
                
                message = await self.websocket.recv()
                
                # Parse binary tick format (Zerodha sends binary)
                if isinstance(message, bytes):
                    tick = self._parse_binary_tick(message)
                    if tick:
                        await self._handle_tick(tick)
                
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed")
                self.data_feed_healthy = False
                await self._handle_reconnection()
            except Exception as e:
                logger.error(f"Receive loop error: {e}")
                await asyncio.sleep(1)
    
    def _parse_binary_tick(self, data: bytes) -> Optional[Dict[str, Any]]:
        """Parse Zerodha binary tick format"""
        try:
            # Zerodha binary format: token(4) + ltp(4) + bid(4) + ask(4) + volume(4)
            # This is simplified; actual format is more complex
            if len(data) < 20:
                return None
            
            token = struct.unpack('>I', data[0:4])[0]
            ltp = struct.unpack('>f', data[4:8])[0]
            bid = struct.unpack('>f', data[8:12])[0]
            ask = struct.unpack('>f', data[12:16])[0]
            volume = struct.unpack('>I', data[16:20])[0]
            
            return {
                'token': token,
                'ltp': ltp,
                'bid': bid,
                'ask': ask,
                'volume': volume,
                'timestamp': datetime.utcnow()
            }
        except Exception as e:
            logger.error(f"Failed to parse tick: {e}")
            return None
    
    async def _handle_tick(self, tick: Dict[str, Any]):
        """Process incoming tick"""
        self.last_heartbeat = datetime.utcnow()
        
        token = tick['token']
        if token in self.tick_callbacks:
            try:
                await self.tick_callbacks[token](tick)
            except Exception as e:
                logger.error(f"Tick callback error: {e}")
    
    async def _heartbeat_monitor(self):
        """Monitor for data feed staleness"""
        while True:
            await asyncio.sleep(5)  # Check every 5 seconds
            
            time_since_last_tick = (
                datetime.utcnow() - self.last_heartbeat
            ).total_seconds()
            
            if time_since_last_tick > 10:  # No data for 10 seconds
                logger.warning(
                    f"Data feed stale for {time_since_last_tick}s, "
                    "market may be closed or connection lost"
                )
                self.data_feed_healthy = False
            elif time_since_last_tick < 10 and not self.data_feed_healthy:
                logger.info("Data feed recovered")
                self.data_feed_healthy = True
    
    async def _handle_reconnection(self):
        """Reconnect with exponential backoff"""
        if self.reconnect_attempt >= len(self.RECONNECT_BACKOFF):
            logger.critical("Max reconnection attempts reached")
            return
        
        backoff = self.RECONNECT_BACKOFF[self.reconnect_attempt]
        logger.info(f"Reconnecting in {backoff}s (attempt {self.reconnect_attempt + 1})")
        
        await asyncio.sleep(backoff)
        self.reconnect_attempt += 1
        
        try:
            await self.connect()
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            await self._handle_reconnection()
    
    async def disconnect(self):
        """Gracefully close connection"""
        if self.websocket:
            await self.websocket.close()
            logger.info("WebSocket disconnected")
```

## 24.2.2 Handling Data Gaps, Reconnections, and Stale Data

```python
from collections import deque
import numpy as np

class DataGapHandler:
    """Manages data quality and consistency"""
    
    def __init__(self, stale_threshold: int = 10):  # seconds
        self.stale_threshold = stale_threshold
        self.last_tick: Dict[str, Dict[str, Any]] = {}
        self.tick_history: Dict[str, deque] = {}
        self.gap_log: Dict[str, list] = {}
    
    def on_tick(self, symbol: str, tick: Dict[str, Any]) -> bool:
        """
        Process tick and detect data anomalies.
        
        Returns:
            bool: True if tick is valid, False if stale/anomalous
        """
        current_time = datetime.utcnow()
        
        # Check for gap
        if symbol in self.last_tick:
            last_tick = self.last_tick[symbol]
            time_gap = (current_time - last_tick['timestamp']).total_seconds()
            
            if time_gap > self.stale_threshold:
                self.gap_log.setdefault(symbol, []).append({
                    'gap_seconds': time_gap,
                    'time': current_time
                })
                logger.warning(f"{symbol}: Data gap of {time_gap}s detected")
                return False
        
        # Check for price jumps (anomaly detection)
        if symbol in self.last_tick:
            prev_price = self.last_tick[symbol]['ltp']
            curr_price = tick['ltp']
            pct_change = abs((curr_price - prev_price) / prev_price)
            
            if pct_change > 0.10:  # > 10% jump
                logger.warning(
                    f"{symbol}: Large price jump {pct_change:.2%} detected"
                )
                # Don't reject, but flag for caution
        
        # Store tick
        self.last_tick[symbol] = {**tick, 'timestamp': current_time}
        
        if symbol not in self.tick_history:
            self.tick_history[symbol] = deque(maxlen=100)
        self.tick_history[symbol].append(tick)
        
        return True
    
    def is_data_stale(self, symbol: str) -> bool:
        """Check if data for symbol is stale"""
        if symbol not in self.last_tick:
            return True
        
        age = (datetime.utcnow() - self.last_tick[symbol]['timestamp']).total_seconds()
        return age > self.stale_threshold
    
    def fill_gap(self, symbol: str) -> Dict[str, Any]:
        """
        Fill data gap with last known price.
        Used when reconnecting during stale data.
        """
        if symbol not in self.last_tick:
            raise ValueError(f"No data history for {symbol}")
        
        last = self.last_tick[symbol]
        
        # Return last known state (mark as stale)
        return {
            **last,
            'is_stale': True,
            'timestamp': datetime.utcnow()
        }
```

## 24.2.3 Real-Time Feature Computation vs Batch

Features must be computed in real-time but cached for efficiency:

```python
from collections import deque
import talib
import numpy as np

class FeatureStore:
    """Real-time feature computation and caching"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.price_windows: Dict[str, deque] = {}
        self.cached_features: Dict[str, Dict[str, float]] = {}
        self.last_update: Dict[str, datetime] = {}
    
    def add_price(self, symbol: str, price: float, timestamp: datetime):
        """Add price and recompute features"""
        if symbol not in self.price_windows:
            self.price_windows[symbol] = deque(maxlen=self.window_size)
        
        self.price_windows[symbol].append(price)
        self.last_update[symbol] = timestamp
        
        # Recompute features incrementally
        if len(self.price_windows[symbol]) >= 10:  # Minimum for some indicators
            self._update_features(symbol)
    
    def _update_features(self, symbol: str):
        """Compute technical features"""
        prices = np.array(list(self.price_windows[symbol]))
        
        if len(prices) < 2:
            return
        
        features = {}
        
        # Momentum indicators
        features['momentum_5'] = (prices[-1] - prices[-5]) / prices[-5] if len(prices) > 5 else 0
        features['momentum_10'] = (prices[-1] - prices[-10]) / prices[-10] if len(prices) > 10 else 0
        
        # Moving averages
        features['sma_10'] = np.mean(prices[-10:]) if len(prices) >= 10 else np.mean(prices)
        features['sma_20'] = np.mean(prices[-20:]) if len(prices) >= 20 else np.mean(prices)
        
        # Volatility (standard deviation of returns)
        returns = np.diff(prices) / prices[:-1]
        features['volatility_10'] = np.std(returns[-10:]) if len(returns) >= 10 else np.std(returns)
        
        # RSI (Relative Strength Index)
        if len(prices) >= 14:
            features['rsi_14'] = self._compute_rsi(prices[-14:])
        
        # MACD
        if len(prices) >= 26:
            macd, signal, hist = self._compute_macd(prices[-26:])
            features['macd'] = macd
            features['macd_signal'] = signal
            features['macd_histogram'] = hist
        
        self.cached_features[symbol] = features
    
    def _compute_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Compute RSI"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        rs = avg_gain / avg_loss if avg_loss > 0 else 0
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _compute_macd(self, prices: np.ndarray) -> tuple:
        """Compute MACD"""
        ema_12 = self._ema(prices, 12)
        ema_26 = self._ema(prices, 26)
        macd = ema_12 - ema_26
        
        macd_signal = self._ema(np.array([macd]), 9)[0]
        macd_histogram = macd - macd_signal
        
        return macd, macd_signal, macd_histogram
    
    def _ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Compute exponential moving average"""
        return prices.ewm(span=period, adjust=False).mean().values
    
    def get_features(self, symbol: str) -> Dict[str, float]:
        """Get cached features for symbol"""
        return self.cached_features.get(symbol, {})
    
    def is_feature_stale(self, symbol: str, max_age_seconds: int = 5) -> bool:
        """Check if features are stale"""
        if symbol not in self.last_update:
            return True
        
        age = (datetime.utcnow() - self.last_update[symbol]).total_seconds()
        return age > max_age_seconds
```

## 24.2.4 Caching and Latency Optimization

```python
from typing import Any
import asyncio

class LatencyOptimizedCache:
    """In-memory cache with latency tracking"""
    
    def __init__(self, max_size: int = 10000):
        self._cache: Dict[str, Any] = {}
        self._access_times: Dict[str, float] = {}
        self._max_size = max_size
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get from cache (async-safe)"""
        async with self._lock:
            if key in self._cache:
                self._access_times[key] = time.time()
                return self._cache[key]
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 300):
        """Set cache with TTL"""
        async with self._lock:
            if len(self._cache) >= self._max_size:
                # Evict least recently accessed
                lru_key = min(self._access_times, key=self._access_times.get)
                del self._cache[lru_key]
                del self._access_times[lru_key]
            
            self._cache[key] = value
            self._access_times[key] = time.time()
    
    async def invalidate(self, key: str):
        """Invalidate cache entry"""
        async with self._lock:
            self._cache.pop(key, None)
            self._access_times.pop(key, None)

class LatencyMonitor:
    """Track component latencies"""
    
    def __init__(self):
        self.latencies: Dict[str, deque] = {}
    
    def record(self, component: str, latency_ms: float):
        """Record latency for component"""
        if component not in self.latencies:
            self.latencies[component] = deque(maxlen=1000)
        
        self.latencies[component].append(latency_ms)
    
    def get_percentile(self, component: str, percentile: int) -> float:
        """Get latency percentile (e.g., p99)"""
        if component not in self.latencies:
            return 0
        
        values = sorted(self.latencies[component])
        idx = int(len(values) * percentile / 100)
        return values[idx] if idx < len(values) else 0
    
    def get_stats(self, component: str) -> Dict[str, float]:
        """Get latency statistics"""
        if component not in self.latencies:
            return {}
        
        values = list(self.latencies[component])
        return {
            'min': min(values),
            'max': max(values),
            'mean': np.mean(values),
            'p50': np.percentile(values, 50),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99),
        }
```

## 24.2.5 Time Synchronization

System time must be synchronized across components for correct sequencing:

```python
import ntplib
from datetime import datetime, timezone

class TimeSync:
    """NTP-based time synchronization"""
    
    def __init__(self, ntp_servers: list = None):
        self.ntp_servers = ntp_servers or ['pool.ntp.org', 'time.nist.gov']
        self.time_offset = 0  # Offset from NTP time
        self.last_sync = None
    
    async def sync(self):
        """Sync with NTP server"""
        for server in self.ntp_servers:
            try:
                client = ntplib.NTPClient()
                response = client.request(server, version=3, timeout=5)
                
                self.time_offset = response.tx_time - datetime.now(timezone.utc).timestamp()
                self.last_sync = datetime.now(timezone.utc)
                
                logger.info(f"Time synced with {server}, offset: {self.time_offset:.3f}s")
                return
            except Exception as e:
                logger.warning(f"NTP sync with {server} failed: {e}")
        
        logger.error("Failed to sync with any NTP server")
    
    def get_current_time(self) -> datetime:
        """Get synchronized current time"""
        return datetime.now(timezone.utc)
    
    def get_timestamp(self) -> float:
        """Get NTP-synchronized timestamp"""
        return datetime.now(timezone.utc).timestamp() + self.time_offset
```

---

# Module 24.3: Logging, Monitoring, and Alerting

## 24.3.1 Structured Logging for Production

```python
import json
import logging
from pythonjsonlogger import jsonlogger
from datetime import datetime
import socket
import os

class ProductionLogger:
    """Structured JSON logging for production systems"""
    
    def __init__(self, log_file: str = "/var/log/trading_system.log"):
        self.log_file = log_file
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure JSON logging"""
        self.logger = logging.getLogger("trading_system")
        self.logger.setLevel(logging.DEBUG)
        
        # File handler with JSON formatter
        file_handler = logging.FileHandler(self.log_file)
        json_formatter = jsonlogger.JsonFormatter(
            fmt='%(timestamp)s %(level)s %(name)s %(message)s %(correlation_id)s'
        )
        file_handler.setFormatter(json_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler for development
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def log_event(self, level: str, event_type: str, message: str,
                  correlation_id: str = None, **extra):
        """
        Log structured event with metadata.
        
        Args:
            level: DEBUG, INFO, WARNING, ERROR, CRITICAL
            event_type: Type of event (SIGNAL, ORDER, FILL, RISK, etc)
            message: Human-readable message
            correlation_id: Request correlation ID for tracing
            **extra: Additional structured data
        """
        
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'hostname': self.hostname,
            'pid': self.pid,
            'correlation_id': correlation_id or 'N/A',
            **extra
        }
        
        log_func = getattr(self.logger, level.lower())
        log_func(message, extra=log_data)
    
    def log_order(self, order_id: str, symbol: str, quantity: int,
                  price: float, side: str, correlation_id: str):
        """Log order placement"""
        self.log_event(
            'info', 'ORDER_PLACED',
            f"Order placed: {symbol} {side} {quantity} @ {price}",
            correlation_id=correlation_id,
            order_id=order_id,
            symbol=symbol,
            quantity=quantity,
            price=price,
            side=side
        )
    
    def log_fill(self, order_id: str, fill_id: str, symbol: str,
                 quantity: int, fill_price: float, commission: float,
                 correlation_id: str):
        """Log order fill"""
        self.log_event(
            'info', 'ORDER_FILLED',
            f"Fill: {symbol} {quantity} @ {fill_price}",
            correlation_id=correlation_id,
            order_id=order_id,
            fill_id=fill_id,
            symbol=symbol,
            quantity=quantity,
            fill_price=fill_price,
            commission=commission
        )
    
    def log_signal(self, symbol: str, signal: int, confidence: float,
                   model_name: str, correlation_id: str):
        """Log ML signal"""
        self.log_event(
            'info', 'SIGNAL_GENERATED',
            f"Signal: {symbol} signal={signal} conf={confidence:.2%} model={model_name}",
            correlation_id=correlation_id,
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            model_name=model_name
        )
    
    def log_risk_violation(self, violation_type: str, current: float,
                          limit: float, correlation_id: str):
        """Log risk limit violation"""
        self.log_event(
            'error', 'RISK_VIOLATION',
            f"Risk violation: {violation_type} current={current} limit={limit}",
            correlation_id=correlation_id,
            violation_type=violation_type,
            current_value=current,
            limit_value=limit
        )
    
    def log_error(self, error_type: str, message: str,
                  correlation_id: str, **context):
        """Log error with context"""
        self.log_event(
            'error', error_type,
            message,
            correlation_id=correlation_id,
            **context
        )

# Example usage:
logger = ProductionLogger()

async def example_logging():
    correlation_id = "req_12345"
    
    logger.log_signal(
        symbol="INFY",
        signal=1,
        confidence=0.72,
        model_name="lstm_v2",
        correlation_id=correlation_id
    )
    
    logger.log_order(
        order_id="ORD_001",
        symbol="INFY",
        quantity=10,
        price=1500.50,
        side="BUY",
        correlation_id=correlation_id
    )
    
    logger.log_fill(
        order_id="ORD_001",
        fill_id="FILL_001",
        symbol="INFY",
        quantity=10,
        fill_price=1500.45,
        commission=10.50,
        correlation_id=correlation_id
    )
```

## 24.3.2 Monitoring Metrics and KPIs

```python
from dataclasses import dataclass
from collections import deque
import numpy as np

@dataclass
class TradingMetrics:
    """Real-time trading metrics"""
    
    # P&L metrics
    realized_pnl: float = 0
    unrealized_pnl: float = 0
    total_pnl: float = 0
    daily_pnl: float = 0
    
    # Trade metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win: float = 0
    avg_loss: float = 0
    win_rate: float = 0
    
    # Risk metrics
    max_drawdown: float = 0
    current_drawdown: float = 0
    sharpe_ratio: float = 0
    sortino_ratio: float = 0
    
    # Position metrics
    total_exposure: float = 0  # Notional value
    leverage: float = 0
    position_count: int = 0
    
    # System metrics
    signal_count: int = 0
    rejection_count: int = 0
    data_gaps: int = 0
    uptime_percent: float = 100
    
    # Latency metrics
    avg_order_latency_ms: float = 0
    p99_order_latency_ms: float = 0

class MetricsCollector:
    """Collects and aggregates metrics"""
    
    def __init__(self, history_size: int = 1440):  # 24 hours of minute data
        self.metrics = TradingMetrics()
        self.pnl_history = deque(maxlen=history_size)
        self.trade_history = deque(maxlen=1000)
        self.hourly_metrics = deque(maxlen=24)
    
    def record_trade(self, entry_price: float, exit_price: float,
                     quantity: int, side: str):
        """Record completed trade"""
        if side == "BUY":
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity
        
        self.trade_history.append({
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'pnl': pnl,
            'timestamp': datetime.utcnow()
        })
        
        self.metrics.total_trades += 1
        if pnl > 0:
            self.metrics.winning_trades += 1
            self.metrics.avg_win = (
                (self.metrics.avg_win * (self.metrics.winning_trades - 1) + pnl) /
                self.metrics.winning_trades
            )
        else:
            self.metrics.losing_trades += 1
            self.metrics.avg_loss = (
                (self.metrics.avg_loss * (self.metrics.losing_trades - 1) + abs(pnl)) /
                self.metrics.losing_trades
            )
        
        self.metrics.total_pnl += pnl
        self.metrics.win_rate = (
            self.metrics.winning_trades / self.metrics.total_trades
            if self.metrics.total_trades > 0 else 0
        )
    
    def update_positions(self, positions: Dict[str, Dict[str, Any]]):
        """Update position metrics"""
        self.metrics.position_count = len(positions)
        self.metrics.total_exposure = sum(
            abs(p['quantity'] * p['current_price']) for p in positions.values()
        )
        
        self.metrics.unrealized_pnl = sum(
            p['unrealized_pnl'] for p in positions.values()
        )
    
    def compute_sharpe_ratio(self, risk_free_rate: float = 0.06) -> float:
        """
        Compute Sharpe ratio.
        
        Sharpe = (mean_return - risk_free_rate) / std_return
        """
        if len(self.pnl_history) < 2:
            return 0
        
        returns = np.diff(list(self.pnl_history)) / np.array(list(self.pnl_history)[:-1])
        returns = returns[~np.isnan(returns)]
        
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        
        annual_return = np.mean(returns) * 252  # Trading days
        annual_vol = np.std(returns) * np.sqrt(252)
        
        sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
        self.metrics.sharpe_ratio = sharpe
        return sharpe
    
    def compute_max_drawdown(self) -> float:
        """
        Compute maximum drawdown.
        
        Max Drawdown = (Peak - Trough) / Peak
        """
        if len(self.pnl_history) < 2:
            return 0
        
        cumulative_pnl = np.cumsum(list(self.pnl_history))
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = (cumulative_pnl - running_max) / np.abs(running_max)
        
        max_dd = np.min(drawdown)
        self.metrics.max_drawdown = abs(max_dd)
        return abs(max_dd)
    
    def get_metrics(self) -> TradingMetrics:
        """Get current metrics snapshot"""
        return self.metrics

```

## 24.3.3 Alert Conditions and Escalation

```python
from enum import Enum
from typing import Callable

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class Alert:
    """Alert message"""
    severity: AlertSeverity
    alert_type: str
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None

class AlertingSystem:
    """Rule-based alert system"""
    
    def __init__(self, logger: ProductionLogger):
        self.logger = logger
        self.alert_callbacks: Dict[str, List[Callable]] = {}
        self.alert_history = deque(maxlen=10000)
        self.active_alerts: Dict[str, Alert] = {}
    
    def register_handler(self, alert_type: str, handler: Callable):
        """Register handler for alert type"""
        if alert_type not in self.alert_callbacks:
            self.alert_callbacks[alert_type] = []
        self.alert_callbacks[alert_type].append(handler)
    
    async def check_position_limits(self, symbol: str, position_qty: int,
                                     max_position: int) -> Optional[Alert]:
        """Alert if position exceeds limit"""
        if abs(position_qty) > max_position:
            alert = Alert(
                severity=AlertSeverity.CRITICAL,
                alert_type="POSITION_LIMIT_EXCEEDED",
                message=f"{symbol}: Position {position_qty} exceeds limit {max_position}",
                correlation_id=None
            )
            await self._trigger_alert(alert)
            return alert
        return None
    
    async def check_daily_loss_limit(self, daily_pnl: float,
                                      max_loss: float) -> Optional[Alert]:
        """Alert if daily loss exceeds limit"""
        if daily_pnl < -max_loss:
            alert = Alert(
                severity=AlertSeverity.CRITICAL,
                alert_type="DAILY_LOSS_EXCEEDED",
                message=f"Daily loss {daily_pnl} exceeds limit {-max_loss}",
                correlation_id=None
            )
            await self._trigger_alert(alert)
            return alert
        return None
    
    async def check_data_feed_health(self, feed_healthy: bool,
                                      max_gap_seconds: int = 10) -> Optional[Alert]:
        """Alert if data feed is unhealthy"""
        if not feed_healthy:
            alert = Alert(
                severity=AlertSeverity.CRITICAL,
                alert_type="DATA_FEED_DOWN",
                message=f"Market data feed down for >{max_gap_seconds}s",
                correlation_id=None
            )
            await self._trigger_alert(alert)
            return alert
        return None
    
    async def check_model_anomaly(self, model_accuracy: float,
                                   min_accuracy: float = 0.45) -> Optional[Alert]:
        """Alert if model accuracy degrades"""
        if model_accuracy < min_accuracy:
            alert = Alert(
                severity=AlertSeverity.WARNING,
                alert_type="MODEL_ANOMALY",
                message=f"Model accuracy {model_accuracy:.1%} below {min_accuracy:.1%}",
                correlation_id=None
            )
            await self._trigger_alert(alert)
            return alert
        return None
    
    async def check_system_latency(self, latency_ms: float,
                                    threshold_ms: float = 500) -> Optional[Alert]:
        """Alert if system latency is high"""
        if latency_ms > threshold_ms:
            alert = Alert(
                severity=AlertSeverity.WARNING,
                alert_type="HIGH_LATENCY",
                message=f"System latency {latency_ms:.0f}ms exceeds {threshold_ms:.0f}ms",
                correlation_id=None
            )
            await self._trigger_alert(alert)
            return alert
        return None
    
    async def _trigger_alert(self, alert: Alert):
        """Trigger alert and call handlers"""
        self.alert_history.append(alert)
        self.active_alerts[alert.alert_type] = alert
        
        # Log alert
        self.logger.log_event(
            alert.severity.value, alert.alert_type,
            alert.message,
            correlation_id=alert.correlation_id
        )
        
        # Call registered handlers
        handlers = self.alert_callbacks.get(alert.alert_type, [])
        for handler in handlers:
            try:
                await handler(alert)
            except Exception as e:
                self.logger.log_error(
                    "ALERT_HANDLER_ERROR",
                    f"Handler failed for {alert.alert_type}: {e}",
                    correlation_id=alert.correlation_id
                )

```

## 24.3.4 Monitoring Dashboard with Streamlit

```python
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import asyncio

class MonitoringDashboard:
    """Real-time monitoring dashboard"""
    
    def __init__(self, metrics_collector: MetricsCollector,
                 db_manager: DatabaseManager, alerting_system: AlertingSystem):
        self.metrics = metrics_collector
        self.db = db_manager
        self.alerts = alerting_system
    
    def render_dashboard(self):
        """Render Streamlit dashboard"""
        st.set_page_config(page_title="Trading System Monitor", layout="wide")
        
        # Header
        st.title("Trading System Real-Time Monitor")
        st.write(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        # Key metrics row
        self._render_metrics_row()
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["P&L", "Positions", "Orders", "Alerts"])
        
        with tab1:
            self._render_pnl_section()
        
        with tab2:
            self._render_positions_section()
        
        with tab3:
            self._render_orders_section()
        
        with tab4:
            self._render_alerts_section()
    
    def _render_metrics_row(self):
        """Render KPI metrics row"""
        metrics = self.metrics.get_metrics()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Total P&L",
                f"₹{metrics.total_pnl:,.0f}",
                delta=f"Daily: ₹{metrics.daily_pnl:,.0f}",
                delta_color="off"
            )
        
        with col2:
            st.metric(
                "Win Rate",
                f"{metrics.win_rate:.1%}",
                f"Trades: {metrics.total_trades}",
                delta_color="off"
            )
        
        with col3:
            st.metric(
                "Max Drawdown",
                f"{metrics.max_drawdown:.1%}",
                delta_color="inverse"
            )
        
        with col4:
            st.metric(
                "Sharpe Ratio",
                f"{metrics.sharpe_ratio:.2f}",
                delta_color="off"
            )
        
        with col5:
            st.metric(
                "Positions",
                metrics.position_count,
                f"Exposure: ₹{metrics.total_exposure:,.0f}",
                delta_color="off"
            )
    
    def _render_pnl_section(self):
        """Render P&L chart"""
        st.subheader("Profit & Loss")
        
        # Cumulative P&L chart
        pnl_data = list(self.metrics.pnl_history)
        cumulative_pnl = np.cumsum(pnl_data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=cumulative_pnl,
            mode='lines',
            name='Cumulative P&L',
            fill='tozeroy',
            line=dict(color='green' if cumulative_pnl[-1] > 0 else 'red')
        ))
        
        fig.update_layout(
            title="Cumulative P&L Over Time",
            xaxis_title="Time",
            yaxis_title="P&L (₹)",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Win/Loss distribution
        col1, col2 = st.columns(2)
        
        with col1:
            trades_df = pd.DataFrame(list(self.metrics.trade_history))
            if not trades_df.empty:
                fig_win_loss = go.Figure(data=[
                    go.Histogram(
                        x=trades_df[trades_df['pnl'] > 0]['pnl'],
                        name='Wins',
                        marker=dict(color='green')
                    ),
                    go.Histogram(
                        x=trades_df[trades_df['pnl'] <= 0]['pnl'],
                        name='Losses',
                        marker=dict(color='red')
                    )
                ])
                
                fig_win_loss.update_layout(
                    title="Trade P&L Distribution",
                    xaxis_title="P&L (₹)",
                    yaxis_title="Count",
                    height=400
                )
                st.plotly_chart(fig_win_loss, use_container_width=True)
        
        with col2:
            metrics = self.metrics.get_metrics()
            
            fig_metrics = go.Figure(data=[
                go.Bar(
                    x=['Avg Win', 'Avg Loss'],
                    y=[metrics.avg_win, metrics.avg_loss],
                    marker=dict(color=['green', 'red'])
                )
            ])
            
            fig_metrics.update_layout(
                title="Average Win/Loss",
                yaxis_title="Amount (₹)",
                height=400
            )
            st.plotly_chart(fig_metrics, use_container_width=True)
    
    def _render_positions_section(self):
        """Render positions table"""
        st.subheader("Current Positions")
        
        positions = asyncio.run(self.db.get_positions())
        
        if positions:
            positions_df = pd.DataFrame([
                {
                    'Symbol': symbol,
                    'Quantity': data['quantity'],
                    'Avg Price': f"₹{data['avg_price']:.2f}",
                    'Current Price': f"₹{data['current_price']:.2f}",
                    'Unrealized P&L': f"₹{data['unrealized_pnl']:.2f}"
                }
                for symbol, data in positions.items()
            ])
            
            st.dataframe(positions_df, use_container_width=True)
        else:
            st.info("No open positions")
    
    def _render_orders_section(self):
        """Render recent orders"""
        st.subheader("Recent Orders")
        
        # Query recent orders from database
        session = self.db.SessionLocal()
        recent_orders = session.query(Order).order_by(
            Order.created_at.desc()
        ).limit(20).all()
        
        if recent_orders:
            orders_df = pd.DataFrame([
                {
                    'Order ID': order.order_id,
                    'Symbol': order.symbol,
                    'Side': order.side,
                    'Quantity': order.quantity,
                    'Price': f"₹{order.price:.2f}",
                    'Status': order.status,
                    'Time': order.created_at.strftime('%H:%M:%S')
                }
                for order in recent_orders
            ])
            
            st.dataframe(orders_df, use_container_width=True)
        else:
            st.info("No recent orders")
        
        session.close()
    
    def _render_alerts_section(self):
        """Render alerts"""
        st.subheader("Active Alerts")
        
        if self.alerts.active_alerts:
            alerts_list = [
                {
                    'Type': alert.alert_type,
                    'Severity': alert.severity.value.upper(),
                    'Message': alert.message,
                    'Time': alert.timestamp.strftime('%H:%M:%S')
                }
                for alert in self.alerts.active_alerts.values()
            ]
            
            alerts_df = pd.DataFrame(alerts_list)
            
            # Color code by severity
            def highlight_severity(row):
                if row['Severity'] == 'CRITICAL':
                    return ['background-color: #ffcccc'] * len(row)
                elif row['Severity'] == 'WARNING':
                    return ['background-color: #ffffcc'] * len(row)
                return [''] * len(row)
            
            st.dataframe(
                alerts_df.style.apply(highlight_severity, axis=1),
                use_container_width=True
            )
        else:
            st.success("No active alerts")

# Run dashboard
if __name__ == "__main__":
    # Initialize components
    logger = ProductionLogger()
    metrics = MetricsCollector()
    db = DatabaseManager("sqlite:///trading.db")
    alerts = AlertingSystem(logger)
    
    dashboard = MonitoringDashboard(metrics, db, alerts)
    dashboard.render_dashboard()
```

## 24.3.5 System Health Checks and SLAs

```python
from dataclasses import dataclass

@dataclass
class SystemSLA:
    """System Service Level Agreements"""
    
    # Uptime SLA
    target_uptime: float = 0.9995  # 99.95% uptime
    
    # Latency SLA
    order_placement_p99_ms: float = 500  # 99th percentile latency
    signal_generation_p99_ms: float = 100
    risk_check_p99_ms: float = 50
    
    # Data SLA
    data_feed_uptime: float = 0.9999  # 99.99%
    max_data_gap_seconds: int = 10
    
    # Accuracy SLA
    min_model_accuracy: float = 0.55
    max_rejection_rate: float = 0.05  # 5% of signals rejected

class HealthCheckSystem:
    """Monitor system health against SLAs"""
    
    def __init__(self, sla: SystemSLA, logger: ProductionLogger):
        self.sla = sla
        self.logger = logger
        self.start_time = datetime.utcnow()
        self.downtime_seconds = 0
    
    def check_system_health(self, metrics: TradingMetrics,
                           latency_stats: Dict[str, Dict[str, float]]) -> Dict[str, bool]:
        """Check all SLA compliance"""
        
        health = {}
        
        # Uptime check
        uptime = self._calculate_uptime()
        health['uptime_sla'] = uptime >= self.sla.target_uptime
        
        # Latency checks
        health['order_latency_sla'] = (
            latency_stats.get('order_placement', {}).get('p99', float('inf')) <
            self.sla.order_placement_p99_ms
        )
        
        health['signal_latency_sla'] = (
            latency_stats.get('signal_generation', {}).get('p99', float('inf')) <
            self.sla.signal_generation_p99_ms
        )
        
        # Data feed check
        health['data_feed_sla'] = (
            metrics.data_gaps < 5  # Less than 5 gaps
        )
        
        # Model accuracy check
        health['model_accuracy_sla'] = (
            self._estimate_model_accuracy() >= self.sla.min_model_accuracy
        )
        
        # Log failures
        for check, passed in health.items():
            if not passed:
                self.logger.log_event(
                    'warning', 'SLA_BREACH',
                    f"SLA check failed: {check}",
                    correlation_id=None,
                    check_name=check
                )
        
        return health
    
    def _calculate_uptime(self) -> float:
        """Calculate uptime percentage"""
        total_time = (datetime.utcnow() - self.start_time).total_seconds()
        uptime = (total_time - self.downtime_seconds) / total_time
        return uptime
    
    def _estimate_model_accuracy(self) -> float:
        """Estimate model accuracy from recent predictions"""
        # In practice, calculate from backtesting or recent performance
        return 0.58

```

---

## Summary: Module 24 Key Takeaways

### Architecture Principles
1. **Event-Driven**: Components communicate via events, not polls
2. **Separation of Concerns**: Each component independently deployable
3. **Fault Tolerance**: Graceful degradation; circuit breakers; retry logic
4. **Observability**: Structured logging, metrics, alerts

### Data Infrastructure
1. **Real-time Ingestion**: WebSocket feeds with reconnection logic
2. **Data Quality**: Gap handling, anomaly detection, staleness monitoring
3. **Feature Caching**: In-memory cache with TTL for low-latency ML
4. **Time Sync**: NTP synchronization for correct event sequencing

### Monitoring & Alerting
1. **Structured Logging**: JSON logs with correlation IDs for tracing
2. **Comprehensive Metrics**: P&L, win rate, Sharpe, drawdown, latency
3. **Rule-Based Alerts**: Position limits, daily loss, data feed health, model anomalies
4. **Real-Time Dashboard**: Streamlit monitoring with key metrics

### Production Readiness
- Type hints on all functions
- Comprehensive error handling
- Circuit breaker patterns
- Database design optimized for trading (indexes, transactions)
- Correlation IDs for distributed tracing
- SLA monitoring and health checks

This architecture enables you to take ML models from notebooks to reliable, observable production systems that survive failures and provide visibility into system behavior.

---

## References and Further Reading

1. **Event-Driven Architecture**: Sam Newman, "Building Microservices"
2. **System Design**: Alex Xu, "System Design Interview"
3. **Monitoring**: Vivek Ratan, "Observability Engineering"
4. **Trading Systems**: Aldridge & Krawciw, "Real-Time Risk"
5. **NSE/Zerodha**: [Kite API Documentation](https://kite.trade)

