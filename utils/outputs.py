from typing import Dict, List, Any, Optional
from strategies import StrategyType
import pandas as pd


class Outputs:
    """
    Class for generating formatted output messages for simulations.
    
    Uses data-driven configuration to eliminate strategy-specific if-else blocks.
    Each strategy type has a defined set of parameters and their display names,
    along with mappings to DataFrame column names for results output.
    """

    def __init__(self):
        """Initialize the Outputs class with strategy configurations."""
        
        # Maps strategy types to their configurable parameters
        # Each entry: (display_name, attribute_name_in_simulation_class)
        self.strategy_parameters: Dict[StrategyType, List[tuple]] = {
            StrategyType.MOMENTUM: [
                ("Tickers", "tickers"),
                ("Band Multipliers", "band_multipliers"),
                ("Trade Frequencies", "trade_frequencies"),
                ("Target Volatilities", "target_volatilities"),
            ],
            StrategyType.MEAN_REVERSION: [
                ("Tickers", "tickers"),
                ("Z-Score Lookbacks", "zscore_lookbacks"),
                ("N Std Uppers", "n_std_uppers"),
                ("N Std Lowers", "n_std_lowers"),
                ("Exit Thresholds", "exit_thresholds"),
                ("Trade Frequencies", "trade_frequencies"),
                ("Target Volatilities", "target_volatilities"),
            ],
            StrategyType.MEAN_REVERSION_RSI: [
                ("Tickers", "tickers"),
                ("RSI Periods", "rsi_periods"),
                ("RSI Oversold Levels", "rsi_oversold_levels"),
                ("RSI Overbought Levels", "rsi_overbought_levels"),
                ("SMA Periods", "sma_periods"),
                ("Trade Frequencies", "trade_frequencies"),
                ("Target Volatilities", "target_volatilities"),
            ],
            StrategyType.STAT_ARB: [
                ("Pairs", "pairs"),
                ("Z-Score Lookbacks", "stat_arb_zscore_lookbacks"),
                ("Entry Thresholds", "entry_thresholds"),
                ("Exit Thresholds", "stat_arb_exit_thresholds"),
                ("Use Dynamic Hedge", "use_dynamic_hedge_options"),
                ("Trade Frequencies", "trade_frequencies"),
                ("Target Volatilities", "target_volatilities"),
            ],
            StrategyType.SUPERVISED: [
                ("Tickers", "tickers"),
                ("Model Paths", "model_paths"),
                ("Window Sizes", "window_sizes"),
                ("Trade Frequencies", "trade_frequencies"),
                ("Target Volatilities", "target_volatilities"),
            ]
        }
        
        # Maps strategy types to their result DataFrame columns
        # Each entry: (display_name, column_name_in_results_df, format_string or None)
        # format_string: None for default, or a format like ".2f", ".4f", ".2%"
        self.strategy_result_columns: Dict[StrategyType, List[tuple]] = {
            StrategyType.MOMENTUM: [
                ("Band Multiplier", "Band Multiplier", None),
            ],
            StrategyType.MEAN_REVERSION: [
                ("Z-Score Lookback", "Z-Score Lookback", None),
                ("N Std Upper", "N Std Upper", None),
                ("N Std Lower", "N Std Lower", None),
                ("Exit Threshold", "Exit Threshold", None),
            ],
            StrategyType.MEAN_REVERSION_RSI: [
                ("RSI Period", "RSI Period", None),
                ("RSI Oversold", "RSI Oversold", None),
                ("RSI Overbought", "RSI Overbought", None),
                ("SMA Period", "SMA Period", None),
            ],
            StrategyType.STAT_ARB: [
                ("Ticker B", "Ticker B", None),
                ("Z-Score Lookback", "Z-Score Lookback", None),
                ("Entry Threshold", "Entry Threshold", None),
                ("Exit Threshold", "Exit Threshold", None),
                ("Use Dynamic Hedge", "Use Dynamic Hedge", None),
            ],
            StrategyType.SUPERVISED: [
                ("Model Path", "Model Path", None),
                ("Window Size", "Window Size", None),
            ],
        }
        
        # Common parameters shown for all strategies
        self.common_result_columns: List[tuple] = [
            ("Trading Frequency", "Trading Frequency", "min"),
            ("Target Volatility", "Target Volatility", ".2%"),
            ("Stop Loss", "Stop Loss (%)", ".2%"),
            ("Take Profit", "Take Profit (%)", ".2%"),
            ("Max Drawdown Limit", "Max Drawdown (%)", ".3%"),
        ]
        
        # Trading activity columns (same for all strategies)
        self.trading_activity_columns: List[tuple] = [
            ("Total Trades", "Total Trades", None),
            ("Buy Trades", "Buy Trades", None),
            ("Sell Trades", "Sell Trades", None),
            ("Stop Losses Hit", "Stop Losses Hit", None),
            ("Take Profits Hit", "Take Profits Hit", None),
        ]
        
        # Strategy performance columns
        self.performance_columns: List[tuple] = [
            ("Total Return", "Total % Return (Strategy)", ".2f%"),
            ("Sharpe Ratio", "Sharpe Ratio (Strategy)", ".4f"),
            ("Sortino Ratio", "Sortino Ratio (Strategy)", ".4f"),
            ("Calmar Ratio", "Calmar Ratio (Strategy)", ".4f"),
            ("Max Drawdown", "Max Drawdown (%)", ".2f%"),
            ("Win Rate", "Win Rate (%)", ".2f%"),
            ("Annualized Volatility", "Annualized Volatility (%)", ".2f%"),
            ("Beta", "Beta", ".4f"),
            ("Alpha (annualized)", "Alpha (annualized %)", ".4f%"),
            ("Correlation", "Correlation", ".4f"),
            ("R-Squared", "R-Squared", ".4f"),
            ("Information Ratio", "Information Ratio", ".4f"),
            ("Treynor Ratio", "Treynor Ratio", ".4f"),
        ]
        
        # Benchmark columns
        self.benchmark_columns: List[tuple] = [
            ("Total Return", "Total % Return (Buy & Hold)", ".2f%"),
            ("Sharpe Ratio", "Sharpe Ratio (Buy & Hold)", ".4f"),
            ("Sortino Ratio", "Sortino Ratio (Buy & Hold)", ".4f"),
            ("Calmar Ratio", "Calmar Ratio (Buy & Hold)", ".4f")
        ]
    
    def log_simulation_parameters(
        self, 
        strategy_type: StrategyType, 
        simulation_instance: Any,
        logger: Any
    ) -> None:
        """
        Log simulation parameters at the start of a simulation run.
        
        Args:
            strategy_type: The type of strategy being simulated
            simulation_instance: The Simulation class instance containing parameter values
            logger: Logger instance to use for output
        """
        params = self.strategy_parameters.get(strategy_type, [])
        
        for display_name, attr_name in params:
            value = getattr(simulation_instance, attr_name, "N/A")
            logger.info(f"{display_name}: {value}")


    def generate_parameters_string(
        self, 
        strategy_type: StrategyType, 
        simulation_instance: Any
    ) -> str:
        """
        Generate a formatted string of simulation parameters.
        
        Args:
            strategy_type: The type of strategy being simulated
            simulation_instance: The Simulation class instance containing parameter values
            
        Returns:
            Formatted string with all parameters
        """
        lines = []
        params = self.strategy_parameters.get(strategy_type, [])
        
        for display_name, attr_name in params:
            value = getattr(simulation_instance, attr_name, "N/A")
            lines.append(f"{display_name}: {value}")
        
        return "\n".join(lines)

    
    def _format_value(self, value: Any, format_spec: Optional[str]) -> str:
        """
        Format a value according to the format specification.
        
        Args:
            value: The value to format
            format_spec: Format specification (e.g., ".2f", ".4f", ".2%", "min", or None)
            
        Returns:
            Formatted string
        """
        if pd.isna(value):
            return "N/A"
        
        if format_spec is None:
            return str(value)
        
        # Handle suffix-style formats (like "min")
        if format_spec == "min":
            return f"{value} min"
        
        # Handle percentage format with % already in column name
        if format_spec.endswith("%"):
            fmt = format_spec[:-1]  # Remove the % from format
            return f"{value:{fmt}}%"
        
        # Standard format
        return f"{value:{format_spec}}"


    def format_strategy_parameters(
        self, 
        strategy_type: StrategyType, 
        row: pd.Series
    ) -> str:
        """
        Format strategy-specific parameters from a results row.
        
        Args:
            strategy_type: The type of strategy
            row: A row from the results DataFrame
            
        Returns:
            Formatted string with strategy parameters
        """
        lines = ["Strategy Parameters:"]
        
        # Strategy-specific columns
        columns = self.strategy_result_columns.get(strategy_type, [])
        for display_name, col_name, fmt in columns:
            if col_name in row.index:
                value = self._format_value(row[col_name], fmt)
                lines.append(f"  • {display_name}: {value}")
        
        # Common columns
        for display_name, col_name, fmt in self.common_result_columns:
            if col_name in row.index:
                value = self._format_value(row[col_name], fmt)
                lines.append(f"  • {display_name}: {value}")
        
        return "\n".join(lines)


    def format_trading_activity(self, row: pd.Series) -> str:
        """
        Format trading activity metrics from a results row.
        
        Args:
            row: A row from the results DataFrame
            
        Returns:
            Formatted string with trading activity
        """
        lines = ["Trading Activity:"]
        
        for display_name, col_name, fmt in self.trading_activity_columns:
            if col_name in row.index:
                value = self._format_value(row[col_name], fmt)
                lines.append(f"  • {display_name}: {value}")
        
        return "\n".join(lines)


    def format_performance(self, row: pd.Series) -> str:
        """
        Format strategy performance metrics from a results row.
        
        Args:
            row: A row from the results DataFrame
            
        Returns:
            Formatted string with performance metrics
        """
        lines = ["Strategy Performance:"]
        
        for display_name, col_name, fmt in self.performance_columns:
            if col_name in row.index:
                value = self._format_value(row[col_name], fmt)
                lines.append(f"  • {display_name}: {value}")
        
        return "\n".join(lines)

    
    def format_benchmark(self, row: pd.Series) -> str:
        """
        Format benchmark comparison metrics from a results row.
        
        Args:
            row: A row from the results DataFrame
            
        Returns:
            Formatted string with benchmark metrics
        """
        lines = ["Benchmark (Buy & Hold):"]
        
        for display_name, col_name, fmt in self.benchmark_columns:
            if col_name in row.index:
                value = self._format_value(row[col_name], fmt)
                lines.append(f"  • {display_name}: {value}")
        
        # Calculate and add alpha
        strategy_return = row.get("Total % Return (Strategy)", 0)
        benchmark_return = row.get("Total % Return (Buy & Hold)", 0)
        if not pd.isna(strategy_return) and not pd.isna(benchmark_return):
            alpha = strategy_return - benchmark_return
            lines.append(f"  • Alpha: {alpha:+.2f}%")
        
        return "\n".join(lines)

    
    def format_ticker_header(
        self, 
        strategy_type: StrategyType, 
        row: pd.Series
    ) -> str:
        """
        Format the ticker header for a results row.
        
        Args:
            strategy_type: The type of strategy
            row: A row from the results DataFrame
            
        Returns:
            Formatted ticker header string
        """
        ticker = row.get("Ticker", "Unknown")
        
        if strategy_type == StrategyType.STAT_ARB:
            ticker_b = row.get("Ticker B", "Unknown")
            return f"{ticker}/{ticker_b}"
        
        return ticker

    
    def format_best_strategy_result(
        self, 
        strategy_type: StrategyType, 
        row: pd.Series
    ) -> str:
        """
        Format a complete best strategy result for a single ticker/pair.
        
        Combines all formatting methods into a single formatted output.
        
        Args:
            strategy_type: The type of strategy
            row: A row from the results DataFrame
            
        Returns:
            Complete formatted string for the result
        """
        sections = [
            "-" * 50,
            self.format_ticker_header(strategy_type, row),
            "-" * 50,
            "",
            self.format_strategy_parameters(strategy_type, row),
            "",
            self.format_trading_activity(row),
            "",
            self.format_performance(row),
            "",
            self.format_benchmark(row),
            "",
        ]
        
        return "\n".join(sections)

    
    def format_all_best_results(
        self, 
        strategy_type: StrategyType, 
        best_df: pd.DataFrame
    ) -> str:
        """
        Format all best strategy results for printing.
        
        Args:
            strategy_type: The type of strategy
            best_df: DataFrame containing best results for each ticker/pair
            
        Returns:
            Complete formatted string for all results
        """
        lines = [
            "",
            "=" * 100,
            f"BEST {strategy_type.value.upper()} STRATEGY CONFIGURATION (by Sharpe Ratio)",
            "=" * 100,
            "",
        ]
        
        for _, row in best_df.iterrows():
            lines.append(self.format_best_strategy_result(strategy_type, row))
        
        lines.append("=" * 100)
        
        return "\n".join(lines)

    
    def print_best_results(
        self, 
        strategy_type: StrategyType, 
        best_df: pd.DataFrame
    ) -> None:
        """
        Print all best strategy results.
        
        Convenience method that formats and prints results.
        
        Args:
            strategy_type: The type of strategy
            best_df: DataFrame containing best results for each ticker/pair
        """
        print(self.format_all_best_results(strategy_type, best_df))

    
    def format_ticker_summary(
        self, 
        ticker: str, 
        ticker_results: pd.DataFrame
    ) -> str:
        """
        Format summary statistics for a single ticker.
        
        Args:
            ticker: Ticker symbol
            ticker_results: DataFrame filtered for this ticker
            
        Returns:
            Formatted summary string
        """
        lines = [f"\n{ticker}:"]
        lines.append(f"  Configurations tested: {len(ticker_results)}")
        
        sharpe_col = "Sharpe Ratio (Strategy)"
        return_col = "Total % Return (Strategy)"
        
        if sharpe_col in ticker_results.columns:
            sharpe_min = ticker_results[sharpe_col].min()
            sharpe_max = ticker_results[sharpe_col].max()
            lines.append(f"  Sharpe Range: {sharpe_min:.2f} to {sharpe_max:.2f}")
        
        if return_col in ticker_results.columns:
            ret_min = ticker_results[return_col].min()
            ret_max = ticker_results[return_col].max()
            lines.append(f"  Return Range: {ret_min:.2f}% to {ret_max:.2f}%")
        
        return "\n".join(lines)

    
    def format_pair_summary(
        self, 
        ticker_a: str, 
        ticker_b: str, 
        pair_results: pd.DataFrame
    ) -> str:
        """
        Format summary statistics for a trading pair.
        
        Args:
            ticker_a: Primary ticker symbol
            ticker_b: Secondary ticker symbol
            pair_results: DataFrame filtered for this pair
            
        Returns:
            Formatted summary string
        """
        return self.format_ticker_summary(f"{ticker_a}/{ticker_b}", pair_results)

    
    def format_simulation_summary(
        self, 
        strategy_type: StrategyType,
        results_df: pd.DataFrame,
        tickers: List[str] = None,
        pairs: List[tuple] = None
    ) -> str:
        """
        Format complete simulation summary.
        
        Args:
            strategy_type: The type of strategy
            results_df: Full results DataFrame
            tickers: List of tickers (for non-stat-arb strategies)
            pairs: List of pairs (for stat-arb strategy)
            
        Returns:
            Formatted summary string
        """
        lines = [
            "",
            "=" * 100,
            f"{strategy_type.value.upper()} Simulation Summary",
            "=" * 100,
        ]
        
        if strategy_type == StrategyType.STAT_ARB and pairs:
            for ticker_a, ticker_b in pairs:
                pair_results = results_df[
                    (results_df["Ticker"] == ticker_a) & 
                    (results_df["Ticker B"] == ticker_b)
                ]
                if not pair_results.empty:
                    lines.append(self.format_pair_summary(ticker_a, ticker_b, pair_results))
        elif tickers:
            for ticker in tickers:
                ticker_results = results_df[results_df["Ticker"] == ticker]
                if not ticker_results.empty:
                    lines.append(self.format_ticker_summary(ticker, ticker_results))
        
        return "\n".join(lines)

    
    def print_simulation_summary(
        self,
        strategy_type: StrategyType,
        results_df: pd.DataFrame,
        tickers: List[str] = None,
        pairs: List[tuple] = None
    ) -> None:
        """
        Print complete simulation summary.
        
        Args:
            strategy_type: The type of strategy
            results_df: Full results DataFrame
            tickers: List of tickers (for non-stat-arb strategies)
            pairs: List of pairs (for stat-arb strategy)
        """
        print(self.format_simulation_summary(strategy_type, results_df, tickers, pairs))
