"""
Market calendar management.

This module provides market calendar functionality including:
    - Market holidays by year
    - Early close dates and times
    - Trading day validation

Classes:
    - MarketCalendar: Manages market holidays and early closes
"""

import pandas as pd
from datetime import time, date
from typing import Dict, List, Set, Optional, Any


class MarketCalendar:
    """
    Market calendar for US equity markets.
    
    Manages market holidays and early close dates for NYSE/NASDAQ.
    
    Example:
        calendar = MarketCalendar()
        
        # Check if a date is a holiday
        if calendar.is_holiday(date(2024, 12, 25)):
            print("Market closed")
        
        # Get early close time
        early_close = calendar.get_early_close_time(date(2024, 12, 24))
        if early_close:
            print(f"Market closes at {early_close}")
        
        # Get all holidays for a year
        holidays_2024 = calendar.get_holidays(2024)
    """
    
    # Standard market hours
    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(16, 0)
    EARLY_CLOSE = time(13, 0)
    
    def __init__(self):
        """Initialize the MarketCalendar with holiday data."""
        self._holidays = self._build_holidays()
        self._early_closes = self._build_early_closes()
    
    def _build_holidays(self) -> Dict[int, List[date]]:
        """
        Build holiday calendar by year.
        
        Returns:
            Dictionary mapping year to list of holiday dates
        """
        return {
            2023: [
                date(2023, 1, 2),    # New Year's Day (observed)
                date(2023, 1, 16),   # MLK Day
                date(2023, 2, 20),   # Presidents Day
                date(2023, 4, 7),    # Good Friday
                date(2023, 5, 29),   # Memorial Day
                date(2023, 6, 19),   # Juneteenth
                date(2023, 7, 4),    # Independence Day
                date(2023, 9, 4),    # Labor Day
                date(2023, 11, 23),  # Thanksgiving
                date(2023, 12, 25),  # Christmas
            ],
            2024: [
                date(2024, 1, 1),    # New Year's Day
                date(2024, 1, 15),   # MLK Day
                date(2024, 2, 19),   # Presidents Day
                date(2024, 3, 29),   # Good Friday
                date(2024, 5, 27),   # Memorial Day
                date(2024, 6, 19),   # Juneteenth
                date(2024, 7, 4),    # Independence Day
                date(2024, 9, 2),    # Labor Day
                date(2024, 11, 28),  # Thanksgiving
                date(2024, 12, 25),  # Christmas
            ],
            2025: [
                date(2025, 1, 1),    # New Year's Day
                date(2025, 1, 20),   # MLK Day
                date(2025, 2, 17),   # Presidents Day
                date(2025, 4, 18),   # Good Friday
                date(2025, 5, 26),   # Memorial Day
                date(2025, 6, 19),   # Juneteenth
                date(2025, 7, 4),    # Independence Day
                date(2025, 9, 1),    # Labor Day
                date(2025, 11, 27),  # Thanksgiving
                date(2025, 12, 25),  # Christmas
            ],
            2026: [
                date(2026, 1, 1),    # New Year's Day
                date(2026, 1, 19),   # MLK Day
                date(2026, 2, 16),   # Presidents Day
                date(2026, 4, 3),    # Good Friday
                date(2026, 5, 25),   # Memorial Day
                date(2026, 6, 19),   # Juneteenth
                date(2026, 7, 3),    # Independence Day (observed)
                date(2026, 9, 7),    # Labor Day
                date(2026, 11, 26),  # Thanksgiving
                date(2026, 12, 25),  # Christmas
            ],
        }
    
    def _build_early_closes(self) -> Dict[int, Dict[date, time]]:
        """
        Build early close calendar by year.
        
        Returns:
            Dictionary mapping year to dict of date -> close time
        """
        return {
            2023: {
                date(2023, 7, 3): self.EARLY_CLOSE,    # Day before Independence Day
                date(2023, 11, 24): self.EARLY_CLOSE,  # Day after Thanksgiving
            },
            2024: {
                date(2024, 7, 3): self.EARLY_CLOSE,    # Day before Independence Day
                date(2024, 11, 29): self.EARLY_CLOSE,  # Day after Thanksgiving
                date(2024, 12, 24): self.EARLY_CLOSE,  # Christmas Eve
            },
            2025: {
                date(2025, 7, 3): self.EARLY_CLOSE,    # Day before Independence Day
                date(2025, 11, 28): self.EARLY_CLOSE,  # Day after Thanksgiving
                date(2025, 12, 24): self.EARLY_CLOSE,  # Christmas Eve
            },
            2026: {
                date(2026, 11, 27): self.EARLY_CLOSE,  # Day after Thanksgiving
                date(2026, 12, 24): self.EARLY_CLOSE,  # Christmas Eve
            },
        }
    
    def get_holidays(self, year: int) -> List[date]:
        """
        Get list of market holidays for a year.
        
        Args:
            year: Calendar year
            
        Returns:
            List of holiday dates
        """
        return self._holidays.get(year, [])
    
    def get_all_holidays(self) -> Set[date]:
        """
        Get all holidays across all years.
        
        Returns:
            Set of all holiday dates
        """
        all_holidays = set()
        for year_holidays in self._holidays.values():
            all_holidays.update(year_holidays)
        return all_holidays
    
    def get_early_closes(self, year: int) -> Dict[date, time]:
        """
        Get early close dates and times for a year.
        
        Args:
            year: Calendar year
            
        Returns:
            Dictionary mapping date to early close time
        """
        return self._early_closes.get(year, {})
    
    def get_all_early_closes(self) -> Dict[date, time]:
        """
        Get all early closes across all years.
        
        Returns:
            Dictionary mapping date to early close time
        """
        all_early_closes = {}
        for year_closes in self._early_closes.values():
            all_early_closes.update(year_closes)
        return all_early_closes
    
    def is_holiday(self, check_date: date) -> bool:
        """
        Check if a date is a market holiday.
        
        Args:
            check_date: Date to check
            
        Returns:
            True if date is a holiday
        """
        year = check_date.year
        return check_date in self._holidays.get(year, [])
    
    def is_early_close(self, check_date: date) -> bool:
        """
        Check if a date is an early close day.
        
        Args:
            check_date: Date to check
            
        Returns:
            True if date is an early close day
        """
        year = check_date.year
        return check_date in self._early_closes.get(year, {})
    
    def get_early_close_time(self, check_date: date) -> Optional[time]:
        """
        Get the early close time for a date.
        
        Args:
            check_date: Date to check
            
        Returns:
            Early close time or None if not an early close day
        """
        year = check_date.year
        return self._early_closes.get(year, {}).get(check_date)
    
    def get_close_time(self, check_date: date) -> time:
        """
        Get the market close time for a date.
        
        Args:
            check_date: Date to check
            
        Returns:
            Close time (early or normal)
        """
        early_close = self.get_early_close_time(check_date)
        return early_close if early_close else self.MARKET_CLOSE
    
    def is_trading_day(self, check_date: date) -> bool:
        """
        Check if a date is a trading day.
        
        Args:
            check_date: Date to check
            
        Returns:
            True if date is a trading day (not weekend or holiday)
        """
        # Check weekend
        if check_date.weekday() >= 5:
            return False
        
        # Check holiday
        if self.is_holiday(check_date):
            return False
        
        return True
    
    def get_trading_minutes(self, check_date: date) -> int:
        """
        Get the number of trading minutes for a date.
        
        Args:
            check_date: Date to check
            
        Returns:
            Number of trading minutes (0 if not a trading day)
        """
        if not self.is_trading_day(check_date):
            return 0
        
        close_time = self.get_close_time(check_date)
        
        # Calculate minutes from open (9:30) to close
        open_minutes = self.MARKET_OPEN.hour * 60 + self.MARKET_OPEN.minute
        close_minutes = close_time.hour * 60 + close_time.minute
        
        return close_minutes - open_minutes
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert calendar to dictionary format.
        
        Returns dictionary compatible with the legacy format used by
        BackTest and Simulation classes.
        
        Returns:
            Dictionary with 'holidays' and 'early_closes' keys
        """
        holidays_dict = {}
        for year, dates in self._holidays.items():
            holidays_dict[str(year)] = {
                'holidays': dates,
                'early_closes': self._early_closes.get(year, {})
            }
        
        return {
            'holidays': holidays_dict,
            'early_closes': self.get_all_early_closes()
        }
    
    def add_holiday(self, holiday_date: date) -> None:
        """
        Add a holiday to the calendar.
        
        Args:
            holiday_date: Date to add as holiday
        """
        year = holiday_date.year
        if year not in self._holidays:
            self._holidays[year] = []
        if holiday_date not in self._holidays[year]:
            self._holidays[year].append(holiday_date)
            self._holidays[year].sort()
    
    def add_early_close(self, early_close_date: date, close_time: time = None) -> None:
        """
        Add an early close to the calendar.
        
        Args:
            early_close_date: Date to add as early close
            close_time: Close time (defaults to 1:00 PM)
        """
        if close_time is None:
            close_time = self.EARLY_CLOSE
        
        year = early_close_date.year
        if year not in self._early_closes:
            self._early_closes[year] = {}
        self._early_closes[year][early_close_date] = close_time


def get_market_calendar() -> Dict[str, Any]:
    """
    Get market calendar in legacy dictionary format.
    
    This is a convenience function for backwards compatibility.
    
    Returns:
        Dictionary with 'holidays' and 'early_closes' keys
    """
    calendar = MarketCalendar()
    return calendar.to_dict()


# Backwards compatibility: Create default calendar instance
_default_calendar = MarketCalendar()


def is_holiday(check_date: date) -> bool:
    """Check if date is a holiday (convenience function)."""
    return _default_calendar.is_holiday(check_date)


def is_early_close(check_date: date) -> bool:
    """Check if date is an early close (convenience function)."""
    return _default_calendar.is_early_close(check_date)


def is_trading_day(check_date: date) -> bool:
    """Check if date is a trading day (convenience function)."""
    return _default_calendar.is_trading_day(check_date)
