"""
Enhanced security for LLM Gateway sessions.
"""

import os
import time
import hmac
import hashlib
import base64
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set, Union
import asyncio

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)


class SessionSecurityManager:
    """
    Manager for session security operations including encryption,
    integrity verification, and security policy enforcement.
    """
    
    def __init__(
        self,
        encryption_key: Optional[bytes] = None,
        hmac_key: Optional[bytes] = None,
        session_max_age_seconds: int = 3600,
        idle_timeout_seconds: int = 300
    ):
        """
        Initialize the session security manager.
        
        Args:
            encryption_key: Key for data encryption (generated if not provided)
            hmac_key: Key for HMAC verification (generated if not provided)
            session_max_age_seconds: Maximum session lifetime
            idle_timeout_seconds: Maximum idle time before session expiration
        """
        # Generate secure keys if not provided
        self.encryption_key = encryption_key or os.urandom(32)  # 256-bit key
        self.hmac_key = hmac_key or os.urandom(32)
        
        # Security policy settings
        self.session_max_age_seconds = session_max_age_seconds
        self.idle_timeout_seconds = idle_timeout_seconds
        
        # Session revocation list
        self.revoked_sessions = set()
        self.revocation_lock = asyncio.Lock()
    
    def encrypt_session_data(self, data: Dict[str, Any]) -> bytes:
        """
        Encrypt session data.
        
        Args:
            data: Session data to encrypt
            
        Returns:
            Encrypted data with authentication tag
        """
        # Generate random IV
        iv = os.urandom(16)
        
        # Convert data to bytes
        data_bytes = json.dumps(data).encode('utf-8')
        
        # Pad data
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(data_bytes) + padder.finalize()
        
        # Encrypt data
        cipher = Cipher(
            algorithms.AES(self.encryption_key),
            modes.CBC(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        # Generate HMAC for integrity verification
        h = hmac.new(self.hmac_key, iv + encrypted_data, hashlib.sha256)
        tag = h.digest()
        
        # Combine IV, encrypted data, and HMAC tag
        result = iv + encrypted_data + tag
        
        return result
    
    def decrypt_session_data(self, encrypted_data: bytes) -> Dict[str, Any]:
        """
        Decrypt session data.
        
        Args:
            encrypted_data: Encrypted session data
            
        Returns:
            Decrypted session data
            
        Raises:
            ValueError: If data integrity verification fails
        """
        # Extract IV, encrypted data, and HMAC tag
        iv = encrypted_data[:16]
        tag = encrypted_data[-32:]
        actual_encrypted_data = encrypted_data[16:-32]
        
        # Verify integrity
        h = hmac.new(self.hmac_key, iv + actual_encrypted_data, hashlib.sha256)
        expected_tag = h.digest()
        
        if not hmac.compare_digest(tag, expected_tag):
            raise ValueError("Session data integrity verification failed")
        
        # Decrypt data
        cipher = Cipher(
            algorithms.AES(self.encryption_key),
            modes.CBC(iv),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(actual_encrypted_data) + decryptor.finalize()
        
        # Unpad data
        unpadder = padding.PKCS7(128).unpadder()
        data_bytes = unpadder.update(padded_data) + unpadder.finalize()
        
        # Convert to dictionary
        return json.loads(data_bytes.decode('utf-8'))
    
    def create_session_token(self, session_id: str, user_data: Dict[str, Any]) -> str:
        """
        Create a secure session token.
        
        Args:
            session_id: Session ID
            user_data: User data to include in token
            
        Returns:
            Secure session token
        """
        now = datetime.utcnow()
        
        # Create token data
        token_data = {
            "session_id": session_id,
            "user_data": user_data,
            "created_at": now.isoformat(),
            "expires_at": (now + timedelta(seconds=self.session_max_age_seconds)).isoformat(),
            "last_activity": now.isoformat()
        }
        
        # Encrypt token data
        encrypted_data = self.encrypt_session_data(token_data)
        
        # Encode for transmission
        return base64.urlsafe_b64encode(encrypted_data).decode('utf-8')
    
    async def validate_session_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validate a session token.
        
        Args:
            token: Session token
            
        Returns:
            Session data if valid, None otherwise
        """
        try:
            # Decode token
            encrypted_data = base64.urlsafe_b64decode(token.encode('utf-8'))
            
            # Decrypt token data
            token_data = self.decrypt_session_data(encrypted_data)
            
            # Check if session is revoked
            async with self.revocation_lock:
                if token_data["session_id"] in self.revoked_sessions:
                    logger.warning(f"Session {token_data['session_id']} has been revoked")
                    return None
            
            # Verify expiration
            now = datetime.utcnow()
            expires_at = datetime.fromisoformat(token_data["expires_at"])
            if now >= expires_at:
                logger.warning(f"Session {token_data['session_id']} has expired")
                return None
            
            # Verify idle timeout
            last_activity = datetime.fromisoformat(token_data["last_activity"])
            idle_time = (now - last_activity).total_seconds()
            if idle_time > self.idle_timeout_seconds:
                logger.warning(f"Session {token_data['session_id']} has exceeded idle timeout")
                return None
            
            # Update last activity
            token_data["last_activity"] = now.isoformat()
            
            return token_data
        
        except Exception as e:
            # Any exception means the token is invalid
            logger.error(f"Error validating session token: {e}")
            return None
    
    async def update_session_token(self, token: str) -> Optional[str]:
        """
        Update a session token's last activity time.
        
        Args:
            token: Session token
            
        Returns:
            Updated session token if valid, None otherwise
        """
        # Validate and get token data
        token_data = await self.validate_session_token(token)
        if not token_data:
            return None
        
        # Update last activity
        token_data["last_activity"] = datetime.utcnow().isoformat()
        
        # Re-encrypt and encode
        encrypted_data = self.encrypt_session_data(token_data)
        return base64.urlsafe_b64encode(encrypted_data).decode('utf-8')
    
    async def revoke_session_token(self, token: str) -> bool:
        """
        Revoke a session token.
        
        Args:
            token: Session token
            
        Returns:
            True if token was valid and is now revoked, False otherwise
        """
        # Validate the token
        token_data = await self.validate_session_token(token)
        if not token_data:
            return False
        
        # Add to revocation list
        session_id = token_data["session_id"]
        async with self.revocation_lock:
            self.revoked_sessions.add(session_id)
            logger.info(f"Session {session_id} has been revoked")
        
        return True
    
    async def cleanup_revoked_sessions(self, max_age_seconds: int = 86400) -> int:
        """
        Clean up old revoked sessions to prevent memory leaks.
        
        Args:
            max_age_seconds: Maximum age of revoked sessions to keep
            
        Returns:
            Number of sessions removed
        """
        # In a real implementation, this would be more sophisticated
        # For now, we'll just clear the revocation list periodically
        async with self.revocation_lock:
            count = len(self.revoked_sessions)
            self.revoked_sessions.clear()
            logger.info(f"Cleaned up {count} revoked sessions")
            return count


class SecureSessionPool:
    """
    Enhanced session pool with security features.
    """
    
    def __init__(
        self,
        security_manager: Optional[SessionSecurityManager] = None,
        max_sessions: int = 100,
        session_ttl_seconds: int = 3600,
        cleanup_interval_seconds: int = 300
    ):
        """
        Initialize the secure session pool.
        
        Args:
            security_manager: Session security manager
            max_sessions: Maximum number of sessions
            session_ttl_seconds: Session time-to-live in seconds
            cleanup_interval_seconds: Interval for cleaning up expired sessions
        """
        self.security_manager = security_manager or SessionSecurityManager(
            session_max_age_seconds=session_ttl_seconds,
            idle_timeout_seconds=session_ttl_seconds // 2
        )
        self.max_sessions = max_sessions
        self.session_ttl_seconds = session_ttl_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds
        
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_lock = asyncio.Lock()
        self.session_counter = 0
        
        # Start cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def create_session(self, user_id: str, metadata: Dict[str, Any] = None) -> str:
        """
        Create a new secure session.
        
        Args:
            user_id: User ID
            metadata: Session metadata
            
        Returns:
            Session token
            
        Raises:
            ValueError: If maximum sessions reached
        """
        async with self.session_lock:
            # Check if we've reached the maximum number of sessions
            if len(self.sessions) >= self.max_sessions:
                # Try to clean up expired sessions first
                await self._cleanup_expired_sessions()
                
                # Check again
                if len(self.sessions) >= self.max_sessions:
                    raise ValueError(f"Maximum sessions reached ({self.max_sessions})")
            
            # Generate session ID
            self.session_counter += 1
            session_id = f"session_{self.session_counter}_{int(time.time())}"
            
            # Create session data
            session_data = {
                "user_id": user_id,
                "metadata": metadata or {},
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Create session token
            token = self.security_manager.create_session_token(session_id, session_data)
            
            # Store session
            self.sessions[session_id] = {
                "token": token,
                "user_id": user_id,
                "metadata": metadata or {},
                "created_at": datetime.utcnow().isoformat(),
                "last_activity": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Created session {session_id} for user {user_id}")
            
            return token
    
    async def validate_session(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validate a session token.
        
        Args:
            token: Session token
            
        Returns:
            Session data if valid, None otherwise
        """
        # Validate token
        token_data = await self.security_manager.validate_session_token(token)
        if not token_data:
            return None
        
        # Get session ID
        session_id = token_data["session_id"]
        
        # Check if session exists
        async with self.session_lock:
            if session_id not in self.sessions:
                logger.warning(f"Session {session_id} not found in session pool")
                return None
            
            # Update last activity
            self.sessions[session_id]["last_activity"] = datetime.utcnow().isoformat()
        
        # Return session data
        return token_data["user_data"]
    
    async def update_session(self, token: str, metadata: Dict[str, Any] = None) -> Optional[str]:
        """
        Update a session.
        
        Args:
            token: Session token
            metadata: New metadata to merge with existing metadata
            
        Returns:
            Updated session token if valid, None otherwise
        """
        # Validate token
        token_data = await self.security_manager.validate_session_token(token)
        if not token_data:
            return None
        
        # Get session ID
        session_id = token_data["session_id"]
        
        # Check if session exists
        async with self.session_lock:
            if session_id not in self.sessions:
                logger.warning(f"Session {session_id} not found in session pool")
                return None
            
            # Update metadata if provided
            if metadata:
                self.sessions[session_id]["metadata"].update(metadata)
                token_data["user_data"]["metadata"].update(metadata)
            
            # Update last activity
            now = datetime.utcnow().isoformat()
            self.sessions[session_id]["last_activity"] = now
            token_data["last_activity"] = now
        
        # Create new token
        encrypted_data = self.security_manager.encrypt_session_data(token_data)
        return base64.urlsafe_b64encode(encrypted_data).decode('utf-8')
    
    async def revoke_session(self, token: str) -> bool:
        """
        Revoke a session.
        
        Args:
            token: Session token
            
        Returns:
            True if session was revoked, False otherwise
        """
        # Validate token
        token_data = await self.security_manager.validate_session_token(token)
        if not token_data:
            return False
        
        # Get session ID
        session_id = token_data["session_id"]
        
        # Remove session
        async with self.session_lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"Revoked session {session_id}")
        
        # Revoke token
        await self.security_manager.revoke_session_token(token)
        
        return True
    
    async def _cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.
        
        Returns:
            Number of sessions removed
        """
        now = datetime.utcnow()
        to_remove = []
        
        # Find expired sessions
        async with self.session_lock:
            for session_id, session in self.sessions.items():
                last_activity = datetime.fromisoformat(session["last_activity"])
                idle_time = (now - last_activity).total_seconds()
                
                if idle_time > self.session_ttl_seconds:
                    to_remove.append(session_id)
            
            # Remove expired sessions
            for session_id in to_remove:
                del self.sessions[session_id]
            
            count = len(to_remove)
            if count > 0:
                logger.info(f"Cleaned up {count} expired sessions")
            
            return count
    
    async def _cleanup_loop(self):
        """Background task to clean up expired sessions."""
        try:
            while True:
                await asyncio.sleep(self.cleanup_interval_seconds)
                await self._cleanup_expired_sessions()
                await self.security_manager.cleanup_revoked_sessions()
        except asyncio.CancelledError:
            logger.info("Session cleanup task cancelled")
        except Exception as e:
            logger.error(f"Error in session cleanup task: {e}")
    
    async def close(self):
        """Close the session pool and clean up resources."""
        if hasattr(self, 'cleanup_task') and self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Session pool closed")
