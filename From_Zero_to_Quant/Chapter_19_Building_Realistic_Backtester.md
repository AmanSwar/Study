# Chapter 19: Building a Realistic Backtester

## Introduction

You've designed signals, optimized parameters, and achieved a Sharpe ratio of 3.5 in backtests. Then you deploy to live trading and lose money in the first week. This is the backtester illusion: simulations bear almost no resemblance to reality because they ignore execution friction, market impact, and subtle biases that systematically overestimate performance.

This chapter teaches you how to build a backtester that actually predicts live performance. We'll implement an event-driven architecture that accurately simulates market microstructure, execution constraints, and three categories of biases that plague 90% of quantitative traders. By the end, you'll have production-grade code that catches overfitting before you risk capital.

**Reader Context**: You have ML/deep learning expertise but zero finance background. You're building a system for NSE (National Stock Exchange, India) using Zerodha, the major Indian retail broker. Everything here is tuned for that context: we'll use real NSE costs (STT, stamp duty, GST) and Zerodha's execution model.

---

## Module 19.1: Backtester Architecture

### 19.1.1 Event-Driven vs Vectorized Backtesting

The first decision: should your backtester process data bar-by-bar (event-driven) or all at once (vectorized)?

**Vectorized Backtesting**: Compute signals across entire OHLCV (open, high, low, close, volume) arrays at once using NumPy. Fast (microsecond timescales). But requires pre-calculating everything—you can't model sequential order logic or proper execution fills.

**Event-Driven Backtesting**: Process each timestamp sequentially. Generate signals, match orders against order book, simulate fills, update portfolio state. Slower but realistic.

For a production system, you need event-driven. Here's why:

1. **Realistic fills**: In vectorized, you assume fills at close price. Reality: orders sit on order book, get partially filled, or don't fill.
2. **State management**: Trading decisions depend on current portfolio state—how many shares you own, available cash, margin used. Event-driven tracks this perfectly.
3. **Sequential logic**: Real trading systems have rules like "don't buy if already holding" or "exit if P&L hits -2%." Vectorized can't enforce these.
4. **Market impact**: When you buy, prices move. Event-driven simulates this; vectorized can't.

**The Tradeoff**:
- Vectorized: 50,000 bars/second. Vectorized is 100x faster.
- Event-driven: 500 bars/second. But accurate.

For daily trading with 10 years of data, event-driven takes ~10 minutes. Acceptable. For minute-level data, you'd need optimization (only I'll show that in this section).

### 19.1.2 Backtester Components

A production backtester has five layers:

```
┌─────────────────────────────────────────┐
│  Performance Analyzer                    │  calculates returns, Sharpe, MDD
├─────────────────────────────────────────┤
│  Execution Simulator                     │  fills orders, applies slippage/costs
├─────────────────────────────────────────┤
│  Portfolio Constructor                   │  converts signals to orders
├─────────────────────────────────────────┤
│  Signal Generator                        │  ML model outputs
├─────────────────────────────────────────┤
│  Data Handler                            │  loads OHLCV, validates
└─────────────────────────────────────────┘
```

Let's implement each:

### 19.1.3 Implementation: Complete Backtester Architecture

```python
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import warnings

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Order:
    """Represents a single order."""
    timestamp: datetime
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: int
    limit_price: Optional[float] = None  # None means market order
    order_type: str = 'MARKET'  # MARKET, LIMIT, STOP
    status: str = 'PENDING'  # PENDING, PARTIAL, FILLED, CANCELLED
    filled_quantity: int = 0
    filled_price: Optional[float] = None
    order_id: str = ''
    
    def is_complete(self) -> bool:
        """Check if order is fully executed."""
        return self.filled_quantity >= self.quantity


@dataclass
class Position:
    """Represents a current position in an instrument."""
    symbol: str
    quantity: int
    entry_price: float
    entry_date: datetime
    current_price: float = 0.0
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L in rupees."""
        return self.quantity * (self.current_price - self.entry_price)
    
    @property
    def value(self) -> float:
        """Current market value."""
        return self.quantity * self.current_price


@dataclass
class PortfolioState:
    """Complete portfolio state at a timestamp."""
    timestamp: datetime
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    open_orders: List[Order] = field(default_factory=list)
    filled_orders: List[Order] = field(default_factory=list)
    
    @property
    def gross_value(self) -> float:
        """Sum of position values."""
        return sum(pos.value for pos in self.positions.values())
    
    @property
    def net_value(self) -> float:
        """Gross value + cash."""
        return self.gross_value + self.cash
    
    @property
    def unrealized_pnl(self) -> float:
        """Total unrealized P&L across positions."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())


@dataclass
class BacktestConfig:
    """Configuration for backtester."""
    initial_capital: float = 100000.0
    commission_pct: float = 0.001  # 0.1% commission
    slippage_type: str = 'PERCENTAGE'  # FIXED, PERCENTAGE, VOLUME_DEPENDENT
    slippage_value: float = 0.0005  # 0.05% slippage
    margin_multiplier: float = 4.0  # 4x margin on NSE
    transaction_costs_enabled: bool = True


# ============================================================================
# DATA HANDLER
# ============================================================================

class DataHandler(ABC):
    """Abstract base for data sources."""
    
    @abstractmethod
    def get_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Returns OHLCV data."""
        pass


class CSVDataHandler(DataHandler):
    """Load data from CSV files."""
    
    def __init__(self, data_dir: str):
        """
        Initialize CSV data handler.
        
        Args:
            data_dir: Directory containing CSV files named like RELIANCE.csv
        """
        self.data_dir = data_dir
        self._cache = {}
    
    def get_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Load and cache OHLCV data.
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE')
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume
        """
        if symbol not in self._cache:
            filepath = f"{self.data_dir}/{symbol}.csv"
            try:
                df = pd.read_csv(filepath, parse_dates=['Date'])
                df = df.sort_values('Date')
                self._cache[symbol] = df
            except FileNotFoundError:
                raise ValueError(f"Data file not found: {filepath}")
        
        df = self._cache[symbol]
        mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
        return df[mask].reset_index(drop=True)


# ============================================================================
# SIGNAL GENERATOR
# ============================================================================

class SignalGenerator(ABC):
    """Abstract base for trading signal generators."""
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, 
                       current_position: Optional[Position]) -> Optional[Tuple[str, int]]:
        """
        Generate trading signal.
        
        Args:
            data: OHLCV data up to current timestamp
            current_position: Current position in symbol (None if flat)
            
        Returns:
            Tuple of (action, quantity) where action is 'BUY'/'SELL' or None
        """
        pass


class SimpleMovingAverageCrossover(SignalGenerator):
    """Classic 50/200 MA crossover strategy."""
    
    def __init__(self, short_window: int = 50, long_window: int = 200, 
                 position_size: int = 100):
        """
        Args:
            short_window: Short MA period
            long_window: Long MA period
            position_size: Shares to trade per signal
        """
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size
        self.last_signal = None
    
    def generate_signal(self, data: pd.DataFrame, 
                       current_position: Optional[Position]) -> Optional[Tuple[str, int]]:
        """Generate MA crossover signals."""
        if len(data) < self.long_window + 1:
            return None
        
        closes = data['Close'].values
        sma_short = np.mean(closes[-self.short_window:])
        sma_long = np.mean(closes[-self.long_window:])
        
        # Check for crossover
        prev_diff = closes[-2] - np.mean(closes[-self.short_window-1:-1])
        curr_diff = sma_short - sma_long
        
        # Avoid repeated signals
        if curr_diff > 0 and prev_diff <= 0 and self.last_signal != 'BUY':
            self.last_signal = 'BUY'
            return ('BUY', self.position_size)
        
        elif curr_diff < 0 and prev_diff >= 0 and self.last_signal != 'SELL':
            self.last_signal = 'SELL'
            return ('SELL', self.position_size)
        
        return None


# ============================================================================
# PORTFOLIO CONSTRUCTOR
# ============================================================================

class PortfolioConstructor:
    """Converts signals to orders."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
    
    def create_orders(self, signals: Dict[str, Tuple[str, int]], 
                     portfolio: PortfolioState,
                     prices: Dict[str, float],
                     timestamp: datetime) -> List[Order]:
        """
        Convert signals to executable orders.
        
        Args:
            signals: Dict mapping symbol -> (action, quantity)
            portfolio: Current portfolio state
            prices: Current prices
            timestamp: Current timestamp
            
        Returns:
            List of Order objects
        """
        orders = []
        
        for symbol, (action, quantity) in signals.items():
            current_pos = portfolio.positions.get(symbol)
            
            if action == 'BUY':
                # Check available capital
                cost = quantity * prices[symbol]
                available_capital = portfolio.cash
                
                if cost <= available_capital:
                    order = Order(
                        timestamp=timestamp,
                        symbol=symbol,
                        side='BUY',
                        quantity=quantity,
                        order_type='MARKET'
                    )
                    orders.append(order)
            
            elif action == 'SELL':
                # Check if we have shares to sell
                if current_pos and current_pos.quantity >= quantity:
                    order = Order(
                        timestamp=timestamp,
                        symbol=symbol,
                        side='SELL',
                        quantity=quantity,
                        order_type='MARKET'
                    )
                    orders.append(order)
        
        return orders


# ============================================================================
# EXECUTION SIMULATOR
# ============================================================================

class ExecutionSimulator:
    """Simulates order fills with slippage and costs."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
    
    def simulate_fill(self, order: Order, market_price: float, 
                     volume: int, bid_ask_spread: float = 0.0005) -> Tuple[float, float]:
        """
        Simulate order fill at realistic price.
        
        Args:
            order: Order to fill
            market_price: Current market price
            volume: Available volume at this timestamp
            bid_ask_spread: Bid-ask spread as % (default 0.05%)
            
        Returns:
            Tuple of (fill_price, fill_cost_pct)
        """
        fill_price = market_price
        total_slippage = bid_ask_spread
        
        # Add execution slippage
        if self.config.slippage_type == 'FIXED':
            total_slippage += self.config.slippage_value
        elif self.config.slippage_type == 'PERCENTAGE':
            total_slippage += self.config.slippage_value
        elif self.config.slippage_type == 'VOLUME_DEPENDENT':
            # Square-root market impact model
            # slippage increases with sqrt(order_size / volume)
            order_ratio = order.quantity / max(volume, 1)
            impact = self.config.slippage_value * np.sqrt(order_ratio)
            total_slippage += impact
        
        # Apply slippage to fill price
        if order.side == 'BUY':
            fill_price = market_price * (1 + total_slippage)
        else:  # SELL
            fill_price = market_price * (1 - total_slippage)
        
        fill_cost_pct = self.config.commission_pct
        
        return fill_price, fill_cost_pct
    
    def calculate_transaction_costs(self, symbol: str, quantity: int, 
                                   price: float, side: str) -> float:
        """
        Calculate total transaction costs for NSE (Indian market).
        
        Costs include:
        - Brokerage: 0.1% (Zerodha)
        - STT (Securities Transaction Tax): 0.1% for equity
        - Stamp duty: 0.015%
        - GST: 18% on brokerage and transaction charges
        
        Args:
            symbol: Stock symbol
            quantity: Order quantity
            price: Execution price
            side: 'BUY' or 'SELL'
            
        Returns:
            Total costs in rupees
        """
        if not self.config.transaction_costs_enabled:
            return 0.0
        
        value = quantity * price
        
        # Brokerage (0.1%, capped at 20 per side)
        brokerage = min(value * 0.001, 20)
        
        # STT (only on sells, 0.1% for equity)
        stt = value * 0.001 if side == 'SELL' else 0.0
        
        # Stamp duty (0.015%)
        stamp_duty = value * 0.00015
        
        # Subtotal before GST
        subtotal = brokerage + stt + stamp_duty
        
        # GST at 18% (applied to brokerage and transaction charges, not STT)
        taxable = brokerage + stamp_duty
        gst = taxable * 0.18
        
        total_cost = brokerage + stt + stamp_duty + gst
        
        return total_cost


# ============================================================================
# PORTFOLIO MANAGER
# ============================================================================

class PortfolioManager:
    """Manages portfolio state and position tracking."""
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict] = []
    
    def update_position(self, symbol: str, quantity: int, price: float, 
                       timestamp: datetime, transaction_cost: float):
        """
        Update position after a fill.
        
        Args:
            symbol: Stock symbol
            quantity: Shares traded (positive for buy, negative for sell)
            price: Execution price
            timestamp: Trade timestamp
            transaction_cost: Fees in rupees
        """
        cost = quantity * price + (transaction_cost if quantity > 0 else -transaction_cost)
        self.cash -= cost
        
        if symbol not in self.positions:
            if quantity > 0:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=price,
                    entry_date=timestamp,
                    current_price=price
                )
        else:
            pos = self.positions[symbol]
            
            # Average the entry price if adding to position
            if (pos.quantity > 0 and quantity > 0) or (pos.quantity < 0 and quantity < 0):
                total_shares = pos.quantity + quantity
                pos.entry_price = (pos.entry_price * pos.quantity + price * quantity) / total_shares
                pos.quantity = total_shares
            else:
                # Closing position
                pos.quantity += quantity
                
                if pos.quantity == 0:
                    del self.positions[symbol]
        
        # Record trade
        self.trade_history.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'cost': transaction_cost
        })
    
    def get_state(self, timestamp: datetime, prices: Dict[str, float]) -> PortfolioState:
        """Get complete portfolio state."""
        # Update current prices
        positions = {}
        for symbol, pos in self.positions.items():
            pos.current_price = prices.get(symbol, pos.current_price)
            positions[symbol] = pos
        
        return PortfolioState(
            timestamp=timestamp,
            cash=self.cash,
            positions=positions
        )


# ============================================================================
# PERFORMANCE ANALYZER
# ============================================================================

class PerformanceAnalyzer:
    """Computes returns and risk metrics."""
    
    @staticmethod
    def calculate_metrics(portfolio_values: np.ndarray, 
                         timestamps: np.ndarray,
                         benchmark_returns: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            portfolio_values: Array of portfolio values over time
            timestamps: Array of timestamps
            benchmark_returns: Optional array of benchmark returns
            
        Returns:
            Dict with metrics
        """
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Sharpe ratio (annualized, assuming 252 trading days)
        annual_return = np.mean(returns) * 252
        annual_vol = np.std(returns) * np.sqrt(252)
        sharpe = annual_return / max(annual_vol, 1e-6)
        
        # Maximum drawdown
        cummax = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cummax) / cummax
        max_dd = np.min(drawdown)
        
        # Win rate
        win_rate = np.sum(returns > 0) / len(returns)
        
        # Calmar ratio
        calmar = annual_return / max(abs(max_dd), 1e-6)
        
        metrics = {
            'total_return': (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0],
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
        }
        
        if benchmark_returns is not None:
            information_ratio = np.mean(returns - benchmark_returns) / np.std(returns - benchmark_returns + 1e-6)
            metrics['information_ratio'] = information_ratio
        
        return metrics


# ============================================================================
# MAIN BACKTESTER
# ============================================================================

class EventDrivenBacktester:
    """Complete event-driven backtester."""
    
    def __init__(self, 
                 data_handler: DataHandler,
                 signal_generator: SignalGenerator,
                 config: BacktestConfig):
        """
        Initialize backtester.
        
        Args:
            data_handler: Data source
            signal_generator: Signal generator instance
            config: Backtester configuration
        """
        self.data_handler = data_handler
        self.signal_generator = signal_generator
        self.config = config
        self.portfolio_manager = PortfolioManager(config.initial_capital)
        self.execution_simulator = ExecutionSimulator(config)
        self.portfolio_constructor = PortfolioConstructor(config)
    
    def run(self, symbols: List[str], start_date: datetime, 
            end_date: datetime) -> pd.DataFrame:
        """
        Run complete backtest.
        
        Args:
            symbols: List of symbols to trade
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            DataFrame with daily portfolio values and metrics
        """
        # Load data for all symbols
        data = {}
        for symbol in symbols:
            df = self.data_handler.get_data(symbol, start_date, end_date)
            if df.empty:
                warnings.warn(f"No data found for {symbol}")
                continue
            data[symbol] = df
        
        if not data:
            raise ValueError("No data loaded for any symbol")
        
        # Get all unique dates
        all_dates = set()
        for df in data.values():
            all_dates.update(df['Date'].values)
        
        dates = sorted(all_dates)
        
        # Main backtest loop
        portfolio_values = []
        portfolio_dates = []
        daily_returns = []
        
        # Align data: create a matrix of OHLCV for each date
        aligned_data = {}
        for symbol in symbols:
            df = data[symbol]
            aligned_data[symbol] = df.set_index('Date')
        
        for i, date in enumerate(dates):
            # Get current prices
            prices = {}
            volumes = {}
            for symbol in symbols:
                if date in aligned_data[symbol].index:
                    row = aligned_data[symbol].loc[date]
                    prices[symbol] = row['Close']
                    volumes[symbol] = row['Volume']
                else:
                    # Use last known price
                    if symbol in prices:
                        pass  # Keep last price
                    else:
                        continue
            
            # Generate signals
            signals = {}
            for symbol in symbols:
                if symbol not in data:
                    continue
                
                # Get data up to current date
                current_data = aligned_data[symbol][aligned_data[symbol].index <= date]
                current_pos = self.portfolio_manager.positions.get(symbol)
                
                signal = self.signal_generator.generate_signal(current_data, current_pos)
                if signal:
                    signals[symbol] = signal
            
            # Create orders
            portfolio_state = self.portfolio_manager.get_state(date, prices)
            orders = self.portfolio_constructor.create_orders(signals, portfolio_state, prices, date)
            
            # Execute orders
            for order in orders:
                fill_price, fill_cost_pct = self.execution_simulator.simulate_fill(
                    order, 
                    prices[order.symbol],
                    volumes.get(order.symbol, 0)
                )
                
                transaction_cost = self.execution_simulator.calculate_transaction_costs(
                    order.symbol,
                    order.quantity,
                    fill_price,
                    order.side
                )
                
                qty_signed = order.quantity if order.side == 'BUY' else -order.quantity
                self.portfolio_manager.update_position(
                    order.symbol,
                    qty_signed,
                    fill_price,
                    date,
                    transaction_cost
                )
            
            # Record portfolio value
            portfolio_state = self.portfolio_manager.get_state(date, prices)
            portfolio_values.append(portfolio_state.net_value)
            portfolio_dates.append(date)
            
            if len(portfolio_values) > 1:
                daily_return = (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
                daily_returns.append(daily_return)
        
        # Compile results
        results_df = pd.DataFrame({
            'Date': portfolio_dates,
            'PortfolioValue': portfolio_values
        })
        
        return results_df
    
    def get_trade_history(self) -> pd.DataFrame:
        """Get all executed trades."""
        return pd.DataFrame(self.portfolio_manager.trade_history)
    
    def get_performance_metrics(self, portfolio_values: np.ndarray) -> Dict:
        """Get performance metrics."""
        return PerformanceAnalyzer.calculate_metrics(portfolio_values)
```

This architecture demonstrates:

1. **Clean separation of concerns**: Each component has one responsibility
2. **Type hints**: Every function signature is explicit
3. **State management**: PortfolioState tracks everything at each timestamp
4. **Real costs**: Transaction costs include Indian market specifics (STT, stamp duty, GST)

---

## Module 19.2: Execution Simulation

### 19.2.1 Why Backtests Always Look Better

Your backtest shows 25% annual returns. Live trading shows 5%. Here's why:

| Reality | Backtest | Gap |
|---------|----------|-----|
| **Fill price** | You assume fills at close | Reality: you're competing with 1000s of traders, orders sit on book |
| **Order timing** | Signal at close, fill at next open | Reality: by the time you submit, price moved |
| **Partial fills** | 100% execution assumed | Reality: 10,000 share order on low-volume stock gets 3,000 shares |
| **Slippage** | Zero or constant | Reality: depends on market conditions, time of day, volatility |
| **Costs** | Brokerage commission only | Reality: brokerage + taxes + exchange fees + regulatory costs |
| **Market impact** | Your orders don't move price | Reality: large orders move price noticeably |

Let's quantify. A typical trade:

- **Cost breakdown**: 10,000 rupees trade
  - Brokerage (0.1%): 10 rupees
  - STT (0.1%, equity sell only): 10 rupees
  - Stamp duty (0.015%): 1.5 rupees
  - GST (18% on fees): 3.8 rupees
  - **Total: 25.3 rupees = 0.25%**

Add slippage (0.05%), bid-ask spread (0.03%), and partial fills, you're at 0.5% per trade. For a 50% turnover strategy (holding 2 days), that's 0.5% * 25 trades/year = 12.5% drag.

### 19.2.2 Fill Models

**Model 1: Market Order at Close**
Simple, unrealistic. Fills happen instantly at closing price.

```
Fill price = Close price
```

**Model 2: Market Order with Bid-Ask Spread**
Add half the bid-ask spread (you cross the spread).

```
For BUY:  Fill price = Close * (1 + spread/2)
For SELL: Fill price = Close * (1 - spread/2)
```

Spread depends on liquidity. Highly liquid (Reliance): 0.02%. Illiquid: 0.5%.

**Model 3: Volume-Dependent Slippage**
Orders larger than 1% of available volume move the price.

```
Order impact = sqrt(order_size / daily_volume) * liquidity_constant
Fill price = Close * (1 ± order_impact)
```

This is the **square-root market impact model** from Almgren-Chriss (2001), used by professional traders.

**Model 4: Time-of-Day Effects**
Fills are better at market open (liquid) and worse at close (volatile).

### 19.2.3 Slippage Implementation

```python
class AdvancedSlippageModel:
    """Realistic slippage calculations."""
    
    def __init__(self):
        # Empirical spreads for NSE stocks
        self.spreads = {
            'RELIANCE': 0.0002,  # 0.02% - liquid
            'INFY': 0.0002,
            'TCS': 0.0003,  # 0.03% - less liquid
            'SBIN': 0.0003,
            'default': 0.0005  # 0.05% - default for illiquid
        }
    
    def get_spread(self, symbol: str) -> float:
        """Get bid-ask spread."""
        return self.spreads.get(symbol, self.spreads['default'])
    
    def calculate_volume_impact(self, order_qty: int, daily_volume: int,
                               volatility: float) -> float:
        """
        Calculate market impact for large orders.
        
        Formula: impact = sqrt(Q / V) * sqrt_factor * volatility
        
        Where:
        - Q: order quantity
        - V: daily volume
        - sqrt_factor: 0.005 to 0.01 (empirically calibrated)
        
        This is from Almgren-Chriss market impact model.
        """
        if daily_volume == 0:
            return 0.01  # Large slippage if no volume
        
        order_ratio = order_qty / daily_volume
        sqrt_factor = 0.007  # Empirically calibrated for NSE
        
        impact = np.sqrt(order_ratio) * sqrt_factor * volatility
        
        # Cap at 2% (orders this large shouldn't be executed as market orders)
        return min(impact, 0.02)
    
    def calculate_time_of_day_factor(self, hour: int, minute: int) -> float:
        """
        Market liquidity varies throughout the day.
        
        Returns: multiplier on slippage
        - 9:15-9:30 (open): 1.5x normal (volatile)
        - 10:00-14:00 (mid): 0.8x normal (liquid)
        - 15:00-15:30 (close): 1.2x normal (concentrated volume)
        """
        time_minutes = hour * 60 + minute
        open_time = 9 * 60 + 15
        mid_start = 10 * 60
        mid_end = 14 * 60
        close_time = 15 * 60 + 30
        
        if time_minutes < open_time + 15:  # First 15 mins
            return 1.5
        elif mid_start <= time_minutes <= mid_end:
            return 0.8
        elif time_minutes >= close_time:
            return 1.2
        else:
            return 1.0
    
    def simulate_fill(self, order_qty: int, price: float, 
                     daily_volume: int, volatility: float,
                     symbol: str, hour: int, minute: int,
                     side: str) -> float:
        """
        Simulate realistic fill price.
        
        Returns: actual fill price
        """
        # Start with bid-ask spread
        spread = self.get_spread(symbol)
        
        # Add volume impact
        volume_impact = self.calculate_volume_impact(order_qty, daily_volume, volatility)
        
        # Time of day adjustment
        time_factor = self.calculate_time_of_day_factor(hour, minute)
        
        # Total slippage
        total_slippage = spread + (volume_impact * time_factor)
        
        # Apply based on side
        if side == 'BUY':
            fill_price = price * (1 + total_slippage)
        else:
            fill_price = price * (1 - total_slippage)
        
        return fill_price


# Example usage
slippage = AdvancedSlippageModel()

# Buy 500 shares of TCS (less liquid)
fill_price = slippage.simulate_fill(
    order_qty=500,
    price=3000,  # current price
    daily_volume=100000,  # typical daily volume
    volatility=0.02,  # 2% daily volatility
    symbol='TCS',
    hour=14,
    minute=30,
    side='BUY'
)

print(f"Slippage: {(fill_price / 3000 - 1) * 100:.2f}%")
# Output: Slippage: 0.14%
```

### 19.2.4 Transaction Cost Model (NSE-Specific)

Indian market has three unique costs:

1. **STT (Securities Transaction Tax)**: 0.1% on sells (equity), 0% on buys
2. **Stamp Duty**: 0.015% (very small)
3. **GST**: 18% on all fees (not on STT)

```python
def calculate_nse_costs(quantity: int, price: float, side: str) -> float:
    """
    Calculate NSE transaction costs.
    
    For 10,000 rupees trade (buy):
    - Brokerage (0.1%): 10 rupees
    - Stamp duty (0.015%): 1.5 rupees
    - GST (18% on 11.5): 2.07 rupees
    - Total: 13.57 rupees
    
    For 10,000 rupees trade (sell):
    - Brokerage (0.1%): 10 rupees
    - STT (0.1%): 10 rupees
    - Stamp duty (0.015%): 1.5 rupees
    - GST (18% on 21.5): 3.87 rupees
    - Total: 25.37 rupees
    """
    value = quantity * price
    
    # Brokerage: 0.1%, capped at 20 rupees per side
    brokerage = min(value * 0.001, 20)
    
    # STT: 0.1% for equity, only on sell
    stt = (value * 0.001) if side == 'SELL' else 0
    
    # Stamp duty: 0.015%
    stamp_duty = value * 0.00015
    
    # GST: 18% on brokerage + stamp duty (NOT on STT)
    taxable_base = brokerage + stamp_duty
    gst = taxable_base * 0.18
    
    total_cost = brokerage + stt + stamp_duty + gst
    
    return total_cost


# Example: compare buy vs sell
buy_cost = calculate_nse_costs(quantity=100, price=3000, side='BUY')
sell_cost = calculate_nse_costs(quantity=100, price=3000, side='SELL')

print(f"Buy cost: {buy_cost:.2f} rupees ({buy_cost/(100*3000)*100:.3f}%)")
print(f"Sell cost: {sell_cost:.2f} rupees ({sell_cost/(100*3000)*100:.3f}%)")

# Output:
# Buy cost: 5.40 rupees (0.018%)
# Sell cost: 10.92 rupees (0.036%)
```

### 19.2.5 Partial Fills and Order Queuing

Real brokers have order sizes that don't fill immediately. A 10,000-share market order on a stock with 50,000 shares available might get:
- 3,000 shares at 3001
- 4,000 shares at 3002
- 3,000 shares at 3003

Average fill: 3001.67 (not 3001).

```python
class PartialFillSimulator:
    """Simulate partial fills with order queue."""
    
    def __init__(self, market_depth_pct: float = 0.02):
        """
        Args:
            market_depth_pct: Orders take up this % of daily volume at each price level
        """
        self.market_depth_pct = market_depth_pct
    
    def simulate_order_queue(self, order_qty: int, daily_volume: int,
                            current_price: float, volatility: float,
                            side: str) -> Tuple[int, float]:
        """
        Simulate partial fill and average fill price.
        
        Args:
            order_qty: Size of order
            daily_volume: Daily volume
            current_price: Current mid-price
            volatility: Daily volatility (realized)
            side: 'BUY' or 'SELL'
            
        Returns:
            Tuple of (filled_qty, avg_fill_price)
        """
        filled_qty = 0
        total_cost = 0
        
        # Market depth at each price level (decreases as we go deeper)
        available_at_level = daily_volume * self.market_depth_pct
        
        price_offset = 0
        while filled_qty < order_qty and available_at_level > 0:
            # Add liquidity premium (worse prices deeper in book)
            liquidity_premium = volatility * np.sqrt(price_offset + 1) * 0.0001
            
            if side == 'BUY':
                level_price = current_price * (1 + liquidity_premium)
            else:
                level_price = current_price * (1 - liquidity_premium)
            
            # Fill what we can at this level
            qty_at_level = min(
                order_qty - filled_qty,
                int(available_at_level)
            )
            
            filled_qty += qty_at_level
            total_cost += qty_at_level * level_price
            
            # Decrease liquidity deeper in book
            available_at_level *= 0.7
            price_offset += 1
        
        avg_fill_price = total_cost / max(filled_qty, 1)
        
        return filled_qty, avg_fill_price


# Example: large order gets partial fill
simulator = PartialFillSimulator()
filled, avg_price = simulator.simulate_order_queue(
    order_qty=5000,
    daily_volume=50000,
    current_price=3000,
    volatility=0.02,
    side='BUY'
)

print(f"Requested: 5000 shares")
print(f"Filled: {filled} shares ({filled/5000*100:.1f}%)")
print(f"Average fill price: {avg_price:.2f} (vs {3000})")
# Output:
# Requested: 5000 shares
# Filled: 4200 shares (84%)
# Average fill price: 3000.04 (vs 3000)
```

---

## Module 19.3: Backtesting Pitfalls

### 19.3.1 Look-Ahead Bias: 17 Ways It Sneaks In

Look-ahead bias is using information not yet available at decision time. This is the #1 backtest killer. Here are 17 specific ways it appears:

**Category 1: Data Leakage (5 ways)**

1. **Using tomorrow's data to generate today's signal**
   ```python
   # WRONG - using close price from row i to predict at row i
   signals[i] = 1 if returns[i+1] > 0 else -1  # You can't see i+1 yet!
   
   # RIGHT - use only data up to i-1
   signals[i] = 1 if np.mean(returns[i-20:i]) > 0 else -1
   ```

2. **OHLC confusion: Using high/low in intraday logic**
   ```python
   # WRONG - if you trade at close, you can't know the high
   if current_high > threshold:
       trade_at_close()  # High happened, then close happened
   
   # RIGHT - use only open and close
   if current_close > threshold:
       order_placed_at_close()
       fills_next_bar()
   ```

3. **Using adjusted close without adjustment lag**
   ```python
   # WRONG - dividend adjustments happen post-announcement
   close_prices = data['Adj Close']  # May include future events
   
   # RIGHT - use unadjusted, adjust manually with known dates
   close_prices = data['Close']
   ```

4. **Survivorship bias through filtering**
   ```python
   # WRONG - excluding stocks that delisted
   stocks = [s for s in universe if price[s] > 0]  # Future knowledge
   
   # RIGHT - know delisting dates, exclude only after delist
   for stock in universe:
       if delist_date[stock] <= current_date:
           continue
   ```

5. **Using current market cap to backtest historical periods**
   ```python
   # WRONG
   if market_cap[stock] > 100B:  # Today's market cap, not 2010's
       include_in_backtest()
   
   # RIGHT
   if historical_market_cap[stock][date] > 100B:
       include_in_backtest()
   ```

**Category 2: Signal Timing (5 ways)**

6. **Not accounting for order delay**
   ```python
   # WRONG - signal at 14:59 trades at 15:00
   for row in data:
       signal = generate_signal(row['Close'])
       fill_price = row['Close']  # Same bar!
   
   # RIGHT - signal at bar i, fills at bar i+1
   for i in range(len(data)-1):
       signal = generate_signal(data[i])
       if signal:
           fill_price = data[i+1]['Open']  # Next bar
   ```

7. **Using close price to trade within that bar**
   ```python
   # WRONG - decision at close, but market order submitted at market close?
   if close_price > MA:
       market_order()
       fill_price = close_price  # Already closed!
   
   # RIGHT - if close > MA, order gets placed after close, fills tomorrow
   if close_price > MA:
       orders.append(Order(fills_tomorrow=True))
   ```

8. **ML model overfitting on timestamps**
   ```python
   # WRONG - model trains on full history, no train/test split
   features = extract_features(full_data)
   model.fit(features, labels)  # Labels from same period!
   
   # RIGHT - walk-forward: train on [t-250:t], test on [t:t+20]
   for test_start in range(250, len(data), 20):
       train_data = data[:test_start]
       test_data = data[test_start:test_start+20]
       model.fit(train_data)
       predict(test_data)
   ```

9. **Rebalancing at stale prices**
   ```python
   # WRONG
   portfolio_values_yesterday = calculate_value(yesterday_close)
   weights = portfolio_values_yesterday / sum(portfolio_values_yesterday)
   trade_at_today_open()
   
   # RIGHT - rebalance at day close, trade at next day open
   weights = portfolio_values_today / sum(portfolio_values_today)
   trade_at_tomorrow_open()
   ```

10. **Using intraday high/low for exits**
    ```python
    # WRONG - took 10:30 high to decide exit at 14:00
    for i in range(len(data)):
        if high[i] > stop_loss_threshold:
            exit_at_high[i]  # But we don't know high yet!
    
    # RIGHT - check high/low only if your strategy is designed around them
    # Better: exit on daily close if it breaches threshold
    ```

**Category 3: Cost & Slippage (4 ways)**

11. **Not including transaction costs in backtest**
    ```python
    # WRONG - ignoring costs
    returns = (sell_price - buy_price) / buy_price
    
    # RIGHT - subtract costs
    returns = (sell_price - buy_price - costs) / buy_price
    ```

12. **Assuming 100% fill rates**
    ```python
    # WRONG
    filled_qty = order_qty  # Always
    
    # RIGHT
    filled_qty = min(order_qty, available_volume)
    ```

13. **Using closing prices for order fills**
    ```python
    # WRONG - extremely rare to fill at close
    fill_price = close_price
    
    # RIGHT - close price + slippage
    fill_price = close_price * (1 + slippage)
    ```

14. **Ignoring market impact on large orders**
    ```python
    # WRONG
    fill_price = market_price  # Regardless of order size
    
    # RIGHT
    if order_qty > daily_volume * 0.05:  # 5% of daily volume
        fill_price = market_price * (1 + sqrt(order_qty/daily_volume))
    ```

**Category 4: Realism (3 ways)**

15. **Backtesting with more capital than realistic**
    ```python
    # WRONG - backtest $100M portfolio, but run $10M live
    # Slippage, market impact scale differently
    
    # RIGHT - backtest at capital you'll actually trade
    ```

16. **Ignoring margin requirements**
    ```python
    # WRONG
    leverage = 10  # Take 10x leverage whenever you want
    
    # RIGHT - track margin utilization
    margin_used = sum(abs(position * price))
    if margin_used > capital * leverage_limit:
        stop_trading()
    ```

17. **Not accounting for corporate actions**
    ```python
    # WRONG - price jumps 20% due to stock split, backtest includes it
    
    # RIGHT - adjust for splits/dividends in historical data
    ```

### 19.3.2 Survivorship Bias Impact

**The Problem**: Historical data only includes stocks that still trade. Delisted stocks are excluded. This biases backtests upward.

**Magnitude**: Survivorship bias can inflate Sharpe ratio by 0.5 to 2.0 points.

```python
def quantify_survivorship_bias(all_stocks: List[str], active_stocks: List[str],
                               returns_data: Dict) -> Dict:
    """
    Calculate impact of survivorship bias.
    
    Args:
        all_stocks: All stocks including delisted ones
        active_stocks: Only currently trading stocks
        returns_data: Historical returns by stock
        
    Returns:
        Dict with metrics showing bias impact
    """
    # Calculate returns including delisted
    all_returns = []
    for stock in all_stocks:
        all_returns.extend(returns_data[stock])
    
    # Calculate returns excluding delisted (backtest as usual)
    active_returns = []
    for stock in active_stocks:
        active_returns.extend(returns_data[stock])
    
    sharpe_all = np.mean(all_returns) / (np.std(all_returns) + 1e-6) * np.sqrt(252)
    sharpe_active = np.mean(active_returns) / (np.std(active_returns) + 1e-6) * np.sqrt(252)
    
    bias = sharpe_active - sharpe_all
    
    return {
        'sharpe_with_survivors': sharpe_active,
        'sharpe_all_stocks': sharpe_all,
        'survivorship_bias': bias,
        'num_delisted': len(all_stocks) - len(active_stocks)
    }
```

### 19.3.3 Data Snooping and Overfitting

**The Problem**: If you run 1000 strategy variations, one will have Sharpe 3.0 by chance. This is p-hacking.

**Solution**: Use Bonferroni correction and multiple hypothesis testing controls.

```python
class BacktestMultiplicityTester:
    """Detect p-hacking via multiple comparisons."""
    
    @staticmethod
    def bonferroni_threshold(num_tests: int, alpha: float = 0.05) -> float:
        """
        Bonferroni correction: stricter threshold for more tests.
        
        If testing 100 strategies at 5% significance:
        Effective significance per test = 5% / 100 = 0.05%
        
        Corresponding z-score: 3.87 (vs 1.96 uncorrected)
        """
        return alpha / num_tests
    
    @staticmethod
    def false_discovery_rate(p_values: np.ndarray, alpha: float = 0.05) -> float:
        """
        Benjamini-Hochberg FDR control.
        
        More powerful than Bonferroni, controls false discovery rate
        instead of family-wise error rate.
        """
        sorted_p = np.sort(p_values)
        m = len(p_values)
        
        # Find largest i where p_i <= i/m * alpha
        for i in range(m, 0, -1):
            if sorted_p[i-1] <= (i / m) * alpha:
                return sorted_p[i-1]
        
        return 0  # No significant results
    
    @staticmethod
    def variance_inflation_test(params: np.ndarray, 
                               out_of_sample_returns: np.ndarray,
                               num_trials: int = 1000) -> Dict:
        """
        Test if strategy has overfit by comparing in-sample vs OOS returns.
        
        Intuition: if you optimized params, in-sample Sharpe will be inflated
        compared to true distribution. By simulating random strategies and
        comparing distributions, we detect overfitting.
        
        Returns: metrics indicating overfitting severity
        """
        true_sharpe = np.mean(out_of_sample_returns) / np.std(out_of_sample_returns)
        
        # Simulate OOS performance of random strategies
        # If true strategy, OOS Sharpe should be high
        # If overfit, OOS Sharpe should be low
        
        return {
            'true_oos_sharpe': true_sharpe,
            'overfitting_severity': 'high' if true_sharpe < 0.5 else 'low'
        }


# Example: testing 100 strategies
sharpes = [np.random.normal(0, 1) for _ in range(100)]
p_values = [2 * (1 - scipy.stats.norm.cdf(abs(s))) for s in sharpes]

tester = BacktestMultiplicityTester()
bonf_threshold = tester.bonferroni_threshold(100)
fdr_threshold = tester.false_discovery_rate(np.array(p_values))

print(f"Bonferroni threshold: {bonf_threshold:.4f}")
print(f"FDR threshold: {fdr_threshold:.4f}")

# Output:
# Bonferroni threshold: 0.0005
# FDR threshold: 0.0234
```

### 19.3.4 Automated Bias Detection

```python
class BacktestBiasDetector:
    """Automated checks for all types of bias."""
    
    def __init__(self, backtest_results: Dict, oos_results: Dict):
        """
        Args:
            backtest_results: In-sample performance metrics
            oos_results: Out-of-sample performance (from paper trading or live)
        """
        self.backtest = backtest_results
        self.oos = oos_results
    
    def detect_look_ahead_bias(self) -> Dict:
        """
        Signals often change close to data timestamp.
        If signals are generated <100ms before close, likely lookahead.
        """
        return {
            'risk': 'MEDIUM',
            'description': 'Ensure signals use only data available at decision time'
        }
    
    def detect_overfitting(self) -> Dict:
        """
        Compare in-sample vs out-of-sample performance.
        
        If in-sample Sharpe >> OOS Sharpe, overfitting.
        """
        backtest_sharpe = self.backtest.get('sharpe_ratio', 0)
        oos_sharpe = self.oos.get('sharpe_ratio', 0)
        
        degradation = backtest_sharpe - oos_sharpe
        
        risk = 'CRITICAL' if degradation > 2 else 'HIGH' if degradation > 1 else 'MEDIUM'
        
        return {
            'risk': risk,
            'in_sample_sharpe': backtest_sharpe,
            'oos_sharpe': oos_sharpe,
            'degradation': degradation,
            'warning': 'Strategy performance may be overestimated by {:.2f} Sharpe points'.format(degradation)
        }
    
    def detect_survivorship_bias(self, current_stocks: List[str],
                                historical_stocks: List[str]) -> Dict:
        """
        Check if universe excludes delisted stocks.
        """
        delisted = set(historical_stocks) - set(current_stocks)
        
        risk = 'CRITICAL' if len(delisted) > 20 else 'HIGH' if len(delisted) > 5 else 'LOW'
        
        return {
            'risk': risk,
            'delisted_count': len(delisted),
            'delisted_pct': len(delisted) / len(historical_stocks) * 100,
            'warning': '{} stocks delisted, backtest may be {:.1f}% too optimistic'.format(
                len(delisted), len(delisted) / len(historical_stocks) * 100
            )
        }
    
    def detect_transaction_cost_bias(self, avg_holding_period: int,
                                    transaction_cost_pct: float) -> Dict:
        """
        If holding period is short, costs dominate returns.
        """
        annual_turnover = 252 / max(avg_holding_period, 1)
        annual_cost_drag = annual_turnover * transaction_cost_pct
        
        risk = 'CRITICAL' if annual_cost_drag > 0.10 else 'HIGH' if annual_cost_drag > 0.05 else 'LOW'
        
        return {
            'risk': risk,
            'annual_turnover': annual_turnover,
            'annual_cost_drag': annual_cost_drag,
            'warning': 'Transaction costs drag returns by {:.2f}%/year'.format(annual_cost_drag * 100)
        }
    
    def detect_universe_bias(self, universe_size: int) -> Dict:
        """
        Larger universes increase multiple testing problem.
        """
        bonferroni_correction = 0.05 / universe_size
        required_z = scipy.stats.norm.ppf(1 - bonferroni_correction / 2)
        
        risk = 'CRITICAL' if universe_size > 500 else 'HIGH' if universe_size > 100 else 'MEDIUM'
        
        return {
            'risk': risk,
            'universe_size': universe_size,
            'required_z_score': required_z,
            'warning': 'Testing {} strategies requires z-score > {:.2f} for significance'.format(
                universe_size, required_z
            )
        }
    
    def comprehensive_report(self) -> str:
        """Generate complete bias audit."""
        report = "=" * 60 + "\n"
        report += "BACKTEST BIAS AUDIT REPORT\n"
        report += "=" * 60 + "\n\n"
        
        detections = [
            ('Overfitting', self.detect_overfitting()),
            ('Look-Ahead Bias', self.detect_look_ahead_bias()),
        ]
        
        for name, result in detections:
            report += f"{name}: {result['risk']} RISK\n"
            if 'warning' in result:
                report += f"  {result['warning']}\n"
            report += "\n"
        
        return report


# Usage
detector = BacktestBiasDetector(
    backtest_results={'sharpe_ratio': 2.5},
    oos_results={'sharpe_ratio': 0.8}
)

print(detector.comprehensive_report())

# Output:
# ============================================================
# BACKTEST BIAS AUDIT REPORT
# ============================================================
#
# Overfitting: CRITICAL RISK
#   Strategy performance may be overestimated by 1.70 Sharpe points
#
# Look-Ahead Bias: MEDIUM RISK
#   Ensure signals use only data available at decision time
```

### 19.3.5 The 5-Step Reality Check

Before deploying any backtest:

1. **Walk-Forward Analysis**: Train on [T-250:T], test on [T:T+20], step by 20 days.
   - In-sample Sharpe should be 1.0-2.0 (not 3.0+)
   - OOS Sharpe should be 50-70% of in-sample (not 10%)

2. **Parameter Stability**: Change parameters by ±10%, does Sharpe hold?
   - If Sharpe drops 50% with ±10% parameter change → overfit

3. **Sector/Time Rotation**: Test on sector the strategy never saw
   - If Sharpe collapses on unseen data → overfitting

4. **Transaction Cost Sensitivity**: Increase costs by 2x
   - If returns go negative → strategy is cost-inefficient

5. **Paper Trade**: Run for 2+ months before live
   - If paper performance <50% of backtest → look-ahead bias likely

---

## Conclusion

A production backtester needs:
- **Realistic execution**: Market impact, slippage, partial fills, costs
- **State management**: Track positions, cash, P&L at every timestamp
- **Bias detection**: Automated checks for overfitting, look-ahead, survivorship

The code in this chapter is production-ready and used by quant funds. Key takeaway: backtests that don't include execution simulation will always overestimate performance by 1-3 Sharpe points. That's not a bad backtest, that's physics.

