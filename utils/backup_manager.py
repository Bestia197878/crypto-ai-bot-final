"""
Database Backup Manager
Handles automated database backups
"""
import shutil
import gzip
from pathlib import Path
from datetime import datetime, timedelta
import schedule
import time
import threading
from typing import List, Optional
from loguru import logger


class DatabaseBackupManager:
    """Manages database backups with rotation"""
    
    def __init__(
        self,
        db_path: str = "data/trading.db",
        backup_dir: str = "data/backups",
        max_backups: int = 10,
        backup_interval_hours: int = 24
    ):
        self.db_path = Path(db_path)
        self.backup_dir = Path(backup_dir)
        self.max_backups = max_backups
        self.backup_interval_hours = backup_interval_hours
        
        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self._stop_event = threading.Event()
        self._thread = None
        
        logger.info(f"DatabaseBackupManager initialized: {db_path} -> {backup_dir}")
    
    def create_backup(self) -> Optional[Path]:
        """Create a new backup of the database"""
        try:
            if not self.db_path.exists():
                logger.error(f"Database file not found: {self.db_path}")
                return None
            
            # Generate backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"trading_backup_{timestamp}.db.gz"
            backup_path = self.backup_dir / backup_filename
            
            # Create compressed backup
            with gzip.open(backup_path, 'wb') as f_out:
                with open(self.db_path, 'rb') as f_in:
                    shutil.copyfileobj(f_in, f_out)
            
            logger.info(f"Backup created: {backup_path}")
            
            # Clean old backups
            self._cleanup_old_backups()
            
            return backup_path
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            return None
    
    def _cleanup_old_backups(self):
        """Remove old backups keeping only max_backups most recent"""
        try:
            backups = sorted(
                self.backup_dir.glob("trading_backup_*.db.gz"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            # Remove old backups
            for backup in backups[self.max_backups:]:
                backup.unlink()
                logger.info(f"Removed old backup: {backup}")
                
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")
    
    def list_backups(self) -> List[Path]:
        """List all available backups"""
        return sorted(
            self.backup_dir.glob("trading_backup_*.db.gz"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
    
    def restore_backup(self, backup_path: Path, target_path: Optional[Path] = None) -> bool:
        """Restore database from backup"""
        try:
            target = target_path or self.db_path
            
            if not backup_path.exists():
                logger.error(f"Backup not found: {backup_path}")
                return False
            
            # Create restore point of current database
            if target.exists():
                restore_point = self.backup_dir / f"pre_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db.gz"
                with gzip.open(restore_point, 'wb') as f_out:
                    with open(target, 'rb') as f_in:
                        shutil.copyfileobj(f_in, f_out)
                logger.info(f"Restore point created: {restore_point}")
            
            # Restore from backup
            with gzip.open(backup_path, 'rb') as f_in:
                with open(target, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            logger.info(f"Database restored from: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False
    
    def start_automated_backups(self):
        """Start automated backup scheduler"""
        if self._thread and self._thread.is_alive():
            logger.warning("Automated backups already running")
            return
        
        self._stop_event.clear()
        
        def run_scheduler():
            schedule.every(self.backup_interval_hours).hours.do(self.create_backup)
            
            while not self._stop_event.is_set():
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        self._thread = threading.Thread(target=run_scheduler, daemon=True)
        self._thread.start()
        
        logger.info(f"Automated backups started (interval: {self.backup_interval_hours}h)")
    
    def stop_automated_backups(self):
        """Stop automated backup scheduler"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Automated backups stopped")


def create_backup():
    """Quick function to create a backup"""
    manager = DatabaseBackupManager()
    backup_path = manager.create_backup()
    if backup_path:
        print(f"✅ Backup created: {backup_path}")
    else:
        print("❌ Backup failed")


def list_backups():
    """Quick function to list backups"""
    manager = DatabaseBackupManager()
    backups = manager.list_backups()
    
    if backups:
        print("📁 Available backups:")
        for i, backup in enumerate(backups, 1):
            size = backup.stat().st_size / 1024  # KB
            print(f"   {i}. {backup.name} ({size:.1f} KB)")
    else:
        print("⚠️ No backups found")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Database Backup Manager')
    parser.add_argument('action', choices=['create', 'list', 'auto'])
    
    args = parser.parse_args()
    
    if args.action == 'create':
        create_backup()
    elif args.action == 'list':
        list_backups()
    elif args.action == 'auto':
        manager = DatabaseBackupManager()
        manager.start_automated_backups()
        print("🔄 Automated backups started. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            manager.stop_automated_backups()
