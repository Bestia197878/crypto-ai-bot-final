"""
Logging utilities
"""
import sys
from pathlib import Path
from loguru import logger
from typing import Optional


def setup_logger(
    log_file: Optional[str] = "logs/trading.log",
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "30 days"
) -> logger:
    """
    Setup logging configuration
    
    Args:
        log_file: Path to log file
        level: Logging level
        rotation: Log rotation size
        retention: Log retention period
    
    Returns:
        Configured logger
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler with color
    logger.add(
        sys.stdout,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        colorize=True
    )
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation=rotation,
            retention=retention,
            compression="zip"
        )
    
    return logger


class TradeLogger:
    """Logger specifically for trading activities"""
    
    def __init__(self, logger_instance: logger):
        self.logger = logger_instance
    
    def log_trade_opened(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ):
        """Log trade opened"""
        self.logger.info(
            f"TRADE OPENED | {symbol} {side.upper()} | "
            f"Qty: {quantity:.6f} @ {price:.2f} | "
            f"SL: {stop_loss:.2f if stop_loss else 'None'} | "
            f"TP: {take_profit:.2f if take_profit else 'None'}"
        )
    
    def log_trade_closed(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_percent: float
    ):
        """Log trade closed"""
        pnl_emoji = "🟢" if pnl > 0 else "🔴"
        self.logger.info(
            f"TRADE CLOSED {pnl_emoji} | {symbol} {side.upper()} | "
            f"Qty: {quantity:.6f} | Entry: {entry_price:.2f} | Exit: {exit_price:.2f} | "
            f"P&L: {pnl:.2f} ({pnl_percent:.2f}%)"
        )
    
    def log_prediction(
        self,
        agent_name: str,
        symbol: str,
        action: str,
        confidence: float,
        price: float
    ):
        """Log agent prediction"""
        self.logger.info(
            f"PREDICTION | {agent_name} | {symbol} | "
            f"Action: {action.upper()} | Confidence: {confidence:.2%} | Price: {price:.2f}"
        )
    
    def log_risk_event(
        self,
        event_type: str,
        message: str,
        severity: str = "WARNING"
    ):
        """Log risk event"""
        self.logger.log(
            severity,
            f"RISK EVENT | {event_type} | {message}"
        )
    
    def log_portfolio_update(
        self,
        total_value: float,
        total_return: float,
        total_return_percent: float,
        open_positions: int
    ):
        """Log portfolio update"""
        self.logger.info(
            f"PORTFOLIO | Value: ${total_value:,.2f} | "
            f"Return: ${total_return:,.2f} ({total_return_percent:.2f}%) | "
            f"Open Positions: {open_positions}"
        )
