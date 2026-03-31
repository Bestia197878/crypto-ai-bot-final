"""
Secure API Key Manager with Rotation Support
Handles secure storage and rotation of exchange API keys
"""
import os
import json
import hashlib
import secrets
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
from loguru import logger


class SecureKeyManager:
    """Manages secure storage and rotation of API keys"""
    
    def __init__(self, keys_file: str = "data/.secure_keys", master_key: Optional[str] = None):
        self.keys_file = Path(keys_file)
        self.keys_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate or use master key for encryption
        self._master_key = master_key or self._generate_master_key()
        self._fernet = Fernet(self._derive_key(self._master_key))
        
        self._keys: Dict[str, Dict] = {}
        self._load_keys()
        
    def _generate_master_key(self) -> str:
        """Generate a master encryption key"""
        # Check environment variable first
        env_key = os.getenv('SECURE_KEY_MASTER')
        if env_key:
            return env_key
            
        # Generate new key and save to file (for development only)
        key = secrets.token_hex(32)
        logger.warning("Generated new master key - set SECURE_KEY_MASTER env var for production")
        return key
    
    def _derive_key(self, master_key: str) -> bytes:
        """Derive Fernet key from master key"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'secure_key_manager_salt_v1',  # In production, use random salt stored separately
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
        return key
    
    def _load_keys(self):
        """Load encrypted keys from file"""
        if not self.keys_file.exists():
            return
            
        try:
            with open(self.keys_file, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self._fernet.decrypt(encrypted_data)
            self._keys = json.loads(decrypted_data.decode())
            logger.info(f"Loaded {len(self._keys)} secure keys")
            
        except Exception as e:
            logger.error(f"Failed to load keys: {e}")
            self._keys = {}
    
    def _save_keys(self):
        """Save encrypted keys to file"""
        try:
            data = json.dumps(self._keys).encode()
            encrypted_data = self._fernet.encrypt(data)
            
            with open(self.keys_file, 'wb') as f:
                f.write(encrypted_data)
                
        except Exception as e:
            logger.error(f"Failed to save keys: {e}")
    
    def store_key(self, exchange: str, api_key: str, secret_key: str, 
                  passphrase: Optional[str] = None):
        """Store API keys securely"""
        key_data = {
            'api_key': api_key,
            'secret_key': secret_key,
            'passphrase': passphrase,
            'created_at': datetime.now().isoformat(),
            'last_rotated': datetime.now().isoformat(),
            'rotation_count': 0
        }
        
        self._keys[exchange] = key_data
        self._save_keys()
        
        # Log key hash for verification (not the actual key)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
        logger.info(f"Stored secure keys for {exchange} (hash: {key_hash}...)")
    
    def get_key(self, exchange: str) -> Optional[Tuple[str, str, Optional[str]]]:
        """Retrieve API keys"""
        key_data = self._keys.get(exchange)
        if not key_data:
            return None
            
        return (
            key_data['api_key'],
            key_data['secret_key'],
            key_data.get('passphrase')
        )
    
    def rotate_key(self, exchange: str, new_api_key: str, new_secret_key: str,
                 new_passphrase: Optional[str] = None) -> bool:
        """Rotate API keys"""
        try:
            if exchange not in self._keys:
                logger.error(f"No existing keys for {exchange} to rotate")
                return False
            
            # Archive old keys
            old_data = self._keys[exchange].copy()
            archive_file = self.keys_file.parent / f".key_archive_{exchange}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Save old keys to archive (also encrypted)
            with open(archive_file, 'wb') as f:
                f.write(self._fernet.encrypt(json.dumps(old_data).encode()))
            
            # Update with new keys
            self._keys[exchange].update({
                'api_key': new_api_key,
                'secret_key': new_secret_key,
                'passphrase': new_passphrase,
                'last_rotated': datetime.now().isoformat(),
                'rotation_count': old_data.get('rotation_count', 0) + 1
            })
            
            self._save_keys()
            
            logger.info(f"Rotated keys for {exchange} (rotation #{self._keys[exchange]['rotation_count']})")
            return True
            
        except Exception as e:
            logger.error(f"Key rotation failed: {e}")
            return False
    
    def should_rotate(self, exchange: str, days: int = 90) -> bool:
        """Check if keys should be rotated"""
        key_data = self._keys.get(exchange)
        if not key_data:
            return False
        
        last_rotated = datetime.fromisoformat(key_data['last_rotated'])
        return datetime.now() - last_rotated > timedelta(days=days)
    
    def get_key_age(self, exchange: str) -> Optional[int]:
        """Get age of keys in days"""
        key_data = self._keys.get(exchange)
        if not key_data:
            return None
        
        created_at = datetime.fromisoformat(key_data['created_at'])
        return (datetime.now() - created_at).days
    
    def list_exchanges(self) -> list:
        """List all exchanges with stored keys"""
        return list(self._keys.keys())
    
    def remove_key(self, exchange: str) -> bool:
        """Remove stored keys for exchange"""
        if exchange in self._keys:
            del self._keys[exchange]
            self._save_keys()
            logger.info(f"Removed keys for {exchange}")
            return True
        return False


# Global key manager instance
_key_manager = None

def get_key_manager() -> SecureKeyManager:
    """Get or create global key manager instance"""
    global _key_manager
    if _key_manager is None:
        _key_manager = SecureKeyManager()
    return _key_manager


def setup_exchange_keys(exchange: str, api_key: str, secret_key: str, 
                       passphrase: Optional[str] = None):
    """Setup API keys for an exchange"""
    manager = get_key_manager()
    manager.store_key(exchange, api_key, secret_key, passphrase)


def get_exchange_keys(exchange: str) -> Optional[Tuple[str, str, Optional[str]]]:
    """Get API keys for an exchange"""
    manager = get_key_manager()
    return manager.get_key(exchange)


def rotate_exchange_keys(exchange: str, new_api_key: str, new_secret_key: str,
                        new_passphrase: Optional[str] = None) -> bool:
    """Rotate API keys for an exchange"""
    manager = get_key_manager()
    return manager.rotate_key(exchange, new_api_key, new_secret_key, new_passphrase)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Secure Key Manager')
    parser.add_argument('action', choices=['setup', 'rotate', 'list', 'check'])
    parser.add_argument('--exchange', type=str, help='Exchange name')
    parser.add_argument('--api-key', type=str, help='API key')
    parser.add_argument('--secret-key', type=str, help='Secret key')
    
    args = parser.parse_args()
    
    manager = get_key_manager()
    
    if args.action == 'setup':
        if args.exchange and args.api_key and args.secret_key:
            manager.store_key(args.exchange, args.api_key, args.secret_key)
            print(f"✅ Keys stored for {args.exchange}")
        else:
            print("❌ Missing required parameters")
    
    elif args.action == 'list':
        exchanges = manager.list_exchanges()
        if exchanges:
            print("📁 Stored exchanges:")
            for ex in exchanges:
                age = manager.get_key_age(ex)
                should_rotate = manager.should_rotate(ex)
                status = "⚠️ ROTATE" if should_rotate else "✅ OK"
                print(f"   {ex}: {age} days old {status}")
        else:
            print("⚠️ No keys stored")
    
    elif args.action == 'check':
        if args.exchange:
            age = manager.get_key_age(args.exchange)
            if age is not None:
                should_rotate = manager.should_rotate(args.exchange)
                print(f"{args.exchange}: {age} days old")
                print("⚠️ Should rotate keys!" if should_rotate else "✅ Keys OK")
            else:
                print(f"❌ No keys for {args.exchange}")
