"""
Monitoring and Alerting System for Crypto Trading AI
Tracks performance and sends alerts for critical events
"""
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass, field
from collections import deque
import json
from pathlib import Path

from loguru import logger
from utils.database import DatabaseManager


@dataclass
class Alert:
    """Alert data structure"""
    timestamp: datetime
    level: str  # 'info', 'warning', 'critical'
    category: str  # 'performance', 'risk', 'system', 'trading'
    message: str
    data: Dict = field(default_factory=dict)
    acknowledged: bool = False


class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self, max_history: int = 1000):
        self.alerts: deque = deque(maxlen=max_history)
        self.handlers: List[Callable[[Alert], None]] = []
        self._lock = threading.Lock()
        
    def add_handler(self, handler: Callable[[Alert], None]):
        """Add alert handler"""
        self.handlers.append(handler)
        
    def send_alert(self, level: str, category: str, message: str, data: Dict = None):
        """Send new alert"""
        alert = Alert(
            timestamp=datetime.now(),
            level=level,
            category=category,
            message=message,
            data=data or {}
        )
        
        with self._lock:
            self.alerts.append(alert)
        
        # Log alert
        log_func = getattr(logger, level, logger.info)
        log_func(f"[{category.upper()}] {message}")
        
        # Notify handlers
        for handler in self.handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
    
    def get_unacknowledged(self) -> List[Alert]:
        """Get unacknowledged alerts"""
        return [a for a in self.alerts if not a.acknowledged]
    
    def acknowledge(self, alert: Alert):
        """Mark alert as acknowledged"""
        alert.acknowledged = True


class PerformanceMonitor:
    """Monitors trading performance metrics"""
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        alert_manager: AlertManager,
        check_interval_seconds: int = 300  # 5 minutes
    ):
        self.db = db_manager
        self.alerts = alert_manager
        self.check_interval = check_interval_seconds
        
        self._stop_event = threading.Event()
        self._thread = None
        
        # Thresholds
        self.thresholds = {
            'max_drawdown_percent': 10.0,
            'min_win_rate': 40.0,
            'max_daily_loss': 500.0,
            'min_profit_factor': 1.0,
            'consecutive_losses': 5
        }
        
    def check_performance(self):
        """Check current performance metrics"""
        try:
            # Get recent trades
            trades = self.db.get_trades(limit=100)
            if not trades:
                return
            
            # Calculate metrics
            closed_trades = [t for t in trades if t.get('status') == 'CLOSED']
            if not closed_trades:
                return
            
            pnl_values = [t.get('pnl', 0) or 0 for t in closed_trades]
            winning = [p for p in pnl_values if p > 0]
            losing = [p for p in pnl_values if p <= 0]
            
            total_pnl = sum(pnl_values)
            win_rate = len(winning) / len(pnl_values) * 100 if pnl_values else 0
            
            # Check drawdown (simplified)
            max_pnl = max(pnl_values) if pnl_values else 0
            current_drawdown = ((max_pnl - total_pnl) / 10000) * 100 if max_pnl > 0 else 0
            
            # Check thresholds and send alerts
            if current_drawdown > self.thresholds['max_drawdown_percent']:
                self.alerts.send_alert(
                    'critical',
                    'risk',
                    f"High drawdown detected: {current_drawdown:.1f}%",
                    {'drawdown': current_drawdown, 'threshold': self.thresholds['max_drawdown_percent']}
                )
            
            if win_rate < self.thresholds['min_win_rate']:
                self.alerts.send_alert(
                    'warning',
                    'performance',
                    f"Low win rate: {win_rate:.1f}%",
                    {'win_rate': win_rate, 'threshold': self.thresholds['min_win_rate']}
                )
            
            # Check daily loss
            today = datetime.now().date()
            daily_pnl = sum(
                t.get('pnl', 0) or 0 
                for t in closed_trades 
                if t.get('exit_time') and datetime.fromisoformat(t['exit_time'].replace('Z', '+00:00')).date() == today
            )
            
            if daily_pnl < -self.thresholds['max_daily_loss']:
                self.alerts.send_alert(
                    'critical',
                    'risk',
                    f"Daily loss limit exceeded: ${abs(daily_pnl):.2f}",
                    {'daily_loss': daily_pnl, 'limit': self.thresholds['max_daily_loss']}
                )
                
        except Exception as e:
            logger.error(f"Performance check failed: {e}")
    
    def start_monitoring(self):
        """Start performance monitoring"""
        if self._thread and self._thread.is_alive():
            return
            
        self._stop_event.clear()
        
        def monitor_loop():
            while not self._stop_event.is_set():
                self.check_performance()
                self._stop_event.wait(self.check_interval)
        
        self._thread = threading.Thread(target=monitor_loop, daemon=True)
        self._thread.start()
        
        logger.info(f"Performance monitoring started (interval: {self.check_interval}s)")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Performance monitoring stopped")


class SystemMonitor:
    """Monitors system health and resources"""
    
    def __init__(self, alert_manager: AlertManager):
        self.alerts = alert_manager
        self._stop_event = threading.Event()
        self._thread = None
        
    def check_system_health(self):
        """Check system resources"""
        try:
            import psutil
            
            # Check CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                self.alerts.send_alert(
                    'warning',
                    'system',
                    f"High CPU usage: {cpu_percent:.1f}%",
                    {'cpu_percent': cpu_percent}
                )
            
            # Check memory
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                self.alerts.send_alert(
                    'warning',
                    'system',
                    f"High memory usage: {memory.percent:.1f}%",
                    {'memory_percent': memory.percent}
                )
            
            # Check disk
            disk = psutil.disk_usage('/')
            if disk.percent > 90:
                self.alerts.send_alert(
                    'critical',
                    'system',
                    f"Low disk space: {disk.percent:.1f}% used",
                    {'disk_percent': disk.percent}
                )
                
        except ImportError:
            logger.warning("psutil not available for system monitoring")
        except Exception as e:
            logger.error(f"System health check failed: {e}")
    
    def start_monitoring(self):
        """Start system monitoring"""
        if self._thread and self._thread.is_alive():
            return
            
        self._stop_event.clear()
        
        def monitor_loop():
            while not self._stop_event.is_set():
                self.check_system_health()
                self._stop_event.wait(60)  # Check every minute
        
        self._thread = threading.Thread(target=monitor_loop, daemon=True)
        self._thread.start()
        
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("System monitoring stopped")


class TradingMonitor:
    """Monitors trading activity"""
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        alert_manager: AlertManager
    ):
        self.db = db_manager
        self.alerts = alert_manager
        self.last_trade_count = 0
        self._stop_event = threading.Event()
        self._thread = None
        
    def check_trading_activity(self):
        """Check for trading activity"""
        try:
            trades = self.db.get_trades(limit=100)
            current_count = len(trades)
            
            # Check if no new trades for extended period
            if current_count == self.last_trade_count:
                # Could indicate trading bot is not working
                pass  # Implement logic based on requirements
            else:
                self.last_trade_count = current_count
                
        except Exception as e:
            logger.error(f"Trading activity check failed: {e}")
    
    def start_monitoring(self):
        """Start trading monitoring"""
        if self._thread and self._thread.is_alive():
            return
            
        self._stop_event.clear()
        
        def monitor_loop():
            while not self._stop_event.is_set():
                self.check_trading_activity()
                self._stop_event.wait(300)  # Check every 5 minutes
        
        self._thread = threading.Thread(target=monitor_loop, daemon=True)
        self._thread.start()
        
        logger.info("Trading monitoring started")
    
    def stop_monitoring(self):
        """Stop trading monitoring"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Trading monitoring stopped")


class MonitoringService:
    """Main monitoring service that coordinates all monitors"""
    
    def __init__(self, db_path: str = "data/trading.db"):
        self.db = DatabaseManager(db_path)
        self.alerts = AlertManager()
        
        # Initialize monitors
        self.performance_monitor = PerformanceMonitor(self.db, self.alerts)
        self.system_monitor = SystemMonitor(self.alerts)
        self.trading_monitor = TradingMonitor(self.db, self.alerts)
        
        # Add default alert handler (console)
        self._add_default_handlers()
        
    def _add_default_handlers(self):
        """Add default alert handlers"""
        def console_handler(alert: Alert):
            icon = {'info': 'ℹ️', 'warning': '⚠️', 'critical': '🚨'}.get(alert.level, '•')
            print(f"{icon} [{alert.level.upper()}] {alert.category}: {alert.message}")
        
        self.alerts.add_handler(console_handler)
    
    def start(self):
        """Start all monitoring"""
        self.performance_monitor.start_monitoring()
        self.system_monitor.start_monitoring()
        self.trading_monitor.start_monitoring()
        
        # Send startup alert
        self.alerts.send_alert('info', 'system', 'Monitoring service started')
        
    def stop(self):
        """Stop all monitoring"""
        self.performance_monitor.stop_monitoring()
        self.system_monitor.stop_monitoring()
        self.trading_monitor.stop_monitoring()
        
        # Send shutdown alert
        self.alerts.send_alert('info', 'system', 'Monitoring service stopped')
    
    def get_status(self) -> Dict:
        """Get monitoring status"""
        return {
            'alerts_total': len(self.alerts.alerts),
            'alerts_unacknowledged': len(self.alerts.get_unacknowledged()),
            'performance_monitoring': self.performance_monitor._thread.is_alive() if self.performance_monitor._thread else False,
            'system_monitoring': self.system_monitor._thread.is_alive() if self.system_monitor._thread else False,
            'trading_monitoring': self.trading_monitor._thread.is_alive() if self.trading_monitor._thread else False
        }


# Global monitoring instance
_monitoring_service = None

def start_monitoring():
    """Start global monitoring service"""
    global _monitoring_service
    if _monitoring_service is None:
        _monitoring_service = MonitoringService()
    _monitoring_service.start()
    return _monitoring_service

def stop_monitoring():
    """Stop global monitoring service"""
    global _monitoring_service
    if _monitoring_service:
        _monitoring_service.stop()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "start":
        service = start_monitoring()
        print("🔄 Monitoring service started. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            stop_monitoring()
            print("\n✅ Monitoring service stopped")
    else:
        print("Usage: python monitoring.py start")
