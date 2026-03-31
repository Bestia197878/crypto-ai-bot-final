"""
Audit Logger - Tracks all system activities for compliance
"""
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
from loguru import logger
import sqlite3


@dataclass
class AuditEvent:
    """Audit event record"""
    timestamp: str
    event_type: str
    user_id: str
    action: str
    resource: str
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    session_id: Optional[str] = None
    result: str = "success"
    hash: str = ""
    
    def __post_init__(self):
        if not self.hash:
            self.hash = self._calculate_hash()
    
    def _calculate_hash(self) -> str:
        """Calculate event hash for integrity"""
        data = f"{self.timestamp}{self.event_type}{self.action}{self.resource}{json.dumps(self.details, sort_keys=True)}"
        return hashlib.sha256(data.encode()).hexdigest()


class AuditLogger:
    """
    Audit logging system for compliance and security
    """
    
    def __init__(self, db_path: str = "data/audit.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
        self.previous_hash = ""
        
        logger.info("AuditLogger initialized")
    
    def _init_database(self):
        """Initialize audit database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                user_id TEXT NOT NULL,
                action TEXT NOT NULL,
                resource TEXT NOT NULL,
                details TEXT,
                ip_address TEXT,
                session_id TEXT,
                result TEXT,
                hash TEXT NOT NULL,
                previous_hash TEXT
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id)
        """)
        
        conn.commit()
        conn.close()
    
    def log(
        self,
        event_type: str,
        user_id: str,
        action: str,
        resource: str,
        details: Dict[str, Any],
        ip_address: Optional[str] = None,
        session_id: Optional[str] = None,
        result: str = "success"
    ) -> AuditEvent:
        """Log an audit event"""
        event = AuditEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            user_id=user_id,
            action=action,
            resource=resource,
            details=details,
            ip_address=ip_address,
            session_id=session_id,
            result=result
        )
        
        # Store in database
        self._store_event(event)
        
        # Update previous hash for chain
        self.previous_hash = event.hash
        
        return event
    
    def _store_event(self, event: AuditEvent):
        """Store event in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO audit_log
            (timestamp, event_type, user_id, action, resource, details,
             ip_address, session_id, result, hash, previous_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event.timestamp,
            event.event_type,
            event.user_id,
            event.action,
            event.resource,
            json.dumps(event.details),
            event.ip_address,
            event.session_id,
            event.result,
            event.hash,
            self.previous_hash
        ))
        
        conn.commit()
        conn.close()
    
    def log_trade(
        self,
        user_id: str,
        trade_action: str,
        symbol: str,
        quantity: float,
        price: float,
        **kwargs
    ):
        """Log trade event"""
        return self.log(
            event_type="TRADE",
            user_id=user_id,
            action=trade_action,
            resource=f"TRADE:{symbol}",
            details={
                "symbol": symbol,
                "quantity": quantity,
                "price": price,
                **kwargs
            }
        )
    
    def log_login(
        self,
        user_id: str,
        ip_address: str,
        success: bool = True
    ):
        """Log login event"""
        return self.log(
            event_type="AUTH",
            user_id=user_id,
            action="LOGIN",
            resource="SESSION",
            details={"ip": ip_address},
            ip_address=ip_address,
            result="success" if success else "failed"
        )
    
    def log_api_call(
        self,
        user_id: str,
        endpoint: str,
        method: str,
        params: Dict,
        response_status: int = 200
    ):
        """Log API call"""
        return self.log(
            event_type="API",
            user_id=user_id,
            action=method,
            resource=endpoint,
            details={
                "params": params,
                "status": response_status
            }
        )
    
    def log_config_change(
        self,
        user_id: str,
        config_key: str,
        old_value: Any,
        new_value: Any
    ):
        """Log configuration change"""
        return self.log(
            event_type="CONFIG",
            user_id=user_id,
            action="UPDATE",
            resource=f"CONFIG:{config_key}",
            details={
                "old_value": old_value,
                "new_value": new_value
            }
        )
    
    def query(
        self,
        user_id: Optional[str] = None,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Query audit log"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM audit_log WHERE 1=1"
        params = []
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        conn.close()
        
        return [dict(row) for row in rows]
    
    def verify_integrity(self) -> bool:
        """Verify audit log integrity"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM audit_log ORDER BY id")
        rows = cursor.fetchall()
        
        previous_hash = ""
        
        for row in rows:
            event = AuditEvent(
                timestamp=row[1],
                event_type=row[2],
                user_id=row[3],
                action=row[4],
                resource=row[5],
                details=json.loads(row[6]) if row[6] else {},
                ip_address=row[7],
                session_id=row[8],
                result=row[9],
                hash=row[10]
            )
            
            # Verify hash chain
            if row[11] != previous_hash:
                logger.error(f"Hash chain broken at event {row[0]}")
                return False
            
            # Verify event hash
            if event.hash != row[10]:
                logger.error(f"Event hash mismatch at event {row[0]}")
                return False
            
            previous_hash = event.hash
        
        conn.close()
        logger.info("Audit log integrity verified")
        return True
    
    def export_to_file(self, filepath: str, start_date: Optional[datetime] = None):
        """Export audit log to file"""
        events = self.query(start_time=start_date)
        
        with open(filepath, 'w') as f:
            for event in events:
                f.write(json.dumps(event, default=str) + '\n')
        
        logger.info(f"Exported {len(events)} audit events to {filepath}")
