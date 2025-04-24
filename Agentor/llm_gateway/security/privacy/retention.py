"""
Data retention policies for the LLM Gateway.

This module provides data retention policies and mechanisms for managing data
lifecycle, including retention periods, data purging, and compliance features.
"""

import time
import logging
import json
from typing import Dict, List, Optional, Set, Union, Any, Tuple, Callable
from enum import Enum
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger(__name__)


class RetentionPeriod(Enum):
    """Retention periods for data."""
    EPHEMERAL = "ephemeral"  # Data is not stored at all
    SESSION = "session"  # Data is retained only for the session
    SHORT_TERM = "short_term"  # Data is retained for a short period (e.g., 24 hours)
    MEDIUM_TERM = "medium_term"  # Data is retained for a medium period (e.g., 30 days)
    LONG_TERM = "long_term"  # Data is retained for a long period (e.g., 1 year)
    PERMANENT = "permanent"  # Data is retained permanently (with user consent)


class DataCategory(Enum):
    """Categories of data for retention purposes."""
    SYSTEM = "system"  # System data (logs, metrics, etc.)
    USER_ACCOUNT = "user_account"  # User account data (username, email, etc.)
    USER_CONTENT = "user_content"  # User-generated content (prompts, messages, etc.)
    GENERATED_CONTENT = "generated_content"  # AI-generated content (responses, etc.)
    ANALYTICS = "analytics"  # Analytics data (usage statistics, etc.)
    SECURITY = "security"  # Security data (auth tokens, etc.)
    BILLING = "billing"  # Billing data (payment info, etc.)
    CUSTOM = "custom"  # Custom data category


class RetentionPolicy:
    """Policy for data retention."""
    
    def __init__(
        self,
        data_category: DataCategory,
        retention_period: RetentionPeriod,
        retention_days: Optional[int] = None,
        purge_method: str = "delete",
        requires_consent: bool = False,
        legal_basis: Optional[str] = None,
        description: Optional[str] = None
    ):
        """
        Initialize retention policy.
        
        Args:
            data_category: Category of data
            retention_period: Retention period
            retention_days: Number of days to retain data (if applicable)
            purge_method: Method for purging data ("delete", "anonymize", "pseudonymize")
            requires_consent: Whether user consent is required
            legal_basis: Legal basis for retention (e.g., "contract", "legitimate_interest")
            description: Description of the policy
        """
        self.data_category = data_category
        self.retention_period = retention_period
        self.retention_days = retention_days
        self.purge_method = purge_method
        self.requires_consent = requires_consent
        self.legal_basis = legal_basis
        self.description = description
        
        # Set default retention days based on period
        if retention_days is None:
            if retention_period == RetentionPeriod.SHORT_TERM:
                self.retention_days = 1
            elif retention_period == RetentionPeriod.MEDIUM_TERM:
                self.retention_days = 30
            elif retention_period == RetentionPeriod.LONG_TERM:
                self.retention_days = 365
            elif retention_period == RetentionPeriod.PERMANENT:
                self.retention_days = None  # No expiration
            else:
                self.retention_days = 0  # Ephemeral or session
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert policy to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "data_category": self.data_category.value,
            "retention_period": self.retention_period.value,
            "retention_days": self.retention_days,
            "purge_method": self.purge_method,
            "requires_consent": self.requires_consent,
            "legal_basis": self.legal_basis,
            "description": self.description
        }
    
    def is_expired(self, timestamp: float) -> bool:
        """
        Check if data with the given timestamp is expired.
        
        Args:
            timestamp: Data timestamp
            
        Returns:
            True if expired, False otherwise
        """
        if self.retention_period == RetentionPeriod.PERMANENT:
            return False
        
        if self.retention_period == RetentionPeriod.EPHEMERAL:
            return True
        
        if self.retention_period == RetentionPeriod.SESSION:
            # Session data is considered expired after the session ends
            # This is typically handled by the session manager
            return False
        
        if self.retention_days is not None:
            # Calculate expiration time
            expiration_time = timestamp + (self.retention_days * 24 * 60 * 60)
            
            # Check if expired
            return time.time() > expiration_time
        
        return False


class DataRetentionManager:
    """Manager for data retention policies."""
    
    # Default policies
    DEFAULT_POLICIES = {
        DataCategory.SYSTEM: RetentionPolicy(
            data_category=DataCategory.SYSTEM,
            retention_period=RetentionPeriod.MEDIUM_TERM,
            retention_days=90,
            purge_method="delete",
            requires_consent=False,
            legal_basis="legitimate_interest",
            description="System data for operational purposes"
        ),
        DataCategory.USER_ACCOUNT: RetentionPolicy(
            data_category=DataCategory.USER_ACCOUNT,
            retention_period=RetentionPeriod.LONG_TERM,
            retention_days=365,
            purge_method="anonymize",
            requires_consent=True,
            legal_basis="contract",
            description="User account data"
        ),
        DataCategory.USER_CONTENT: RetentionPolicy(
            data_category=DataCategory.USER_CONTENT,
            retention_period=RetentionPeriod.MEDIUM_TERM,
            retention_days=30,
            purge_method="delete",
            requires_consent=True,
            legal_basis="consent",
            description="User-generated content"
        ),
        DataCategory.GENERATED_CONTENT: RetentionPolicy(
            data_category=DataCategory.GENERATED_CONTENT,
            retention_period=RetentionPeriod.MEDIUM_TERM,
            retention_days=30,
            purge_method="delete",
            requires_consent=True,
            legal_basis="consent",
            description="AI-generated content"
        ),
        DataCategory.ANALYTICS: RetentionPolicy(
            data_category=DataCategory.ANALYTICS,
            retention_period=RetentionPeriod.MEDIUM_TERM,
            retention_days=90,
            purge_method="anonymize",
            requires_consent=True,
            legal_basis="legitimate_interest",
            description="Analytics data for service improvement"
        ),
        DataCategory.SECURITY: RetentionPolicy(
            data_category=DataCategory.SECURITY,
            retention_period=RetentionPeriod.MEDIUM_TERM,
            retention_days=90,
            purge_method="delete",
            requires_consent=False,
            legal_basis="legitimate_interest",
            description="Security data for protection against threats"
        ),
        DataCategory.BILLING: RetentionPolicy(
            data_category=DataCategory.BILLING,
            retention_period=RetentionPeriod.LONG_TERM,
            retention_days=730,  # 2 years
            purge_method="anonymize",
            requires_consent=False,
            legal_basis="legal_obligation",
            description="Billing data for financial records"
        )
    }
    
    def __init__(
        self,
        policies: Optional[Dict[DataCategory, RetentionPolicy]] = None,
        enable_default_policies: bool = True,
        data_store: Optional[Dict[str, Dict[str, Any]]] = None,
        purge_handlers: Optional[Dict[str, Callable[[str, Dict[str, Any]], None]]] = None
    ):
        """
        Initialize data retention manager.
        
        Args:
            policies: Dictionary mapping data categories to retention policies
            enable_default_policies: Whether to enable default policies
            data_store: Dictionary for storing data (for demonstration)
            purge_handlers: Dictionary mapping purge methods to handler functions
        """
        self.policies = {}
        
        # Add default policies if enabled
        if enable_default_policies:
            self.policies.update(self.DEFAULT_POLICIES)
        
        # Add custom policies
        if policies:
            self.policies.update(policies)
        
        # Data store (for demonstration)
        self.data_store = data_store or {}
        
        # Purge handlers
        self.purge_handlers = purge_handlers or {
            "delete": self._delete_data,
            "anonymize": self._anonymize_data,
            "pseudonymize": self._pseudonymize_data
        }
        
        # Lock for thread safety
        self.lock = asyncio.Lock()
    
    def get_policy(self, data_category: DataCategory) -> Optional[RetentionPolicy]:
        """
        Get retention policy for a data category.
        
        Args:
            data_category: Data category
            
        Returns:
            Retention policy or None if not found
        """
        return self.policies.get(data_category)
    
    def set_policy(self, policy: RetentionPolicy) -> None:
        """
        Set retention policy for a data category.
        
        Args:
            policy: Retention policy
        """
        self.policies[policy.data_category] = policy
    
    def get_all_policies(self) -> Dict[DataCategory, RetentionPolicy]:
        """
        Get all retention policies.
        
        Returns:
            Dictionary of retention policies
        """
        return self.policies.copy()
    
    async def store_data(
        self,
        data_id: str,
        data: Dict[str, Any],
        data_category: DataCategory,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store data with retention policy.
        
        Args:
            data_id: Data ID
            data: Data to store
            data_category: Data category
            metadata: Additional metadata
        """
        # Get policy for this category
        policy = self.get_policy(data_category)
        if not policy:
            logger.warning(f"No retention policy found for category {data_category.value}")
            return
        
        # Check if ephemeral
        if policy.retention_period == RetentionPeriod.EPHEMERAL:
            logger.info(f"Data {data_id} is ephemeral, not storing")
            return
        
        # Create data entry
        data_entry = {
            "id": data_id,
            "data": data,
            "category": data_category.value,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        # Store data
        async with self.lock:
            self.data_store[data_id] = data_entry
    
    async def get_data(self, data_id: str) -> Optional[Dict[str, Any]]:
        """
        Get stored data.
        
        Args:
            data_id: Data ID
            
        Returns:
            Stored data or None if not found or expired
        """
        async with self.lock:
            # Check if data exists
            if data_id not in self.data_store:
                return None
            
            # Get data entry
            data_entry = self.data_store[data_id]
            
            # Get policy for this category
            category = DataCategory(data_entry["category"])
            policy = self.get_policy(category)
            
            if not policy:
                logger.warning(f"No retention policy found for category {category.value}")
                return data_entry["data"]
            
            # Check if expired
            if policy.is_expired(data_entry["timestamp"]):
                # Purge data
                await self._purge_data(data_id)
                return None
            
            return data_entry["data"]
    
    async def delete_data(self, data_id: str) -> bool:
        """
        Delete stored data.
        
        Args:
            data_id: Data ID
            
        Returns:
            True if deleted, False if not found
        """
        async with self.lock:
            if data_id in self.data_store:
                del self.data_store[data_id]
                return True
            return False
    
    async def purge_expired_data(self) -> int:
        """
        Purge all expired data.
        
        Returns:
            Number of purged data entries
        """
        purged_count = 0
        
        async with self.lock:
            # Get all data IDs
            data_ids = list(self.data_store.keys())
            
            # Check each data entry
            for data_id in data_ids:
                data_entry = self.data_store[data_id]
                
                # Get policy for this category
                category = DataCategory(data_entry["category"])
                policy = self.get_policy(category)
                
                if not policy:
                    logger.warning(f"No retention policy found for category {category.value}")
                    continue
                
                # Check if expired
                if policy.is_expired(data_entry["timestamp"]):
                    # Purge data
                    await self._purge_data(data_id)
                    purged_count += 1
        
        return purged_count
    
    async def _purge_data(self, data_id: str) -> None:
        """
        Purge data using the appropriate method.
        
        Args:
            data_id: Data ID
        """
        # Get data entry
        data_entry = self.data_store.get(data_id)
        if not data_entry:
            return
        
        # Get policy for this category
        category = DataCategory(data_entry["category"])
        policy = self.get_policy(category)
        
        if not policy:
            logger.warning(f"No retention policy found for category {category.value}")
            return
        
        # Get purge method
        purge_method = policy.purge_method
        
        # Get purge handler
        purge_handler = self.purge_handlers.get(purge_method)
        
        if not purge_handler:
            logger.warning(f"No purge handler found for method {purge_method}")
            return
        
        # Call purge handler
        await purge_handler(data_id, data_entry)
    
    async def _delete_data(self, data_id: str, data_entry: Dict[str, Any]) -> None:
        """
        Delete data.
        
        Args:
            data_id: Data ID
            data_entry: Data entry
        """
        # Delete data
        if data_id in self.data_store:
            del self.data_store[data_id]
            logger.info(f"Deleted data {data_id}")
    
    async def _anonymize_data(self, data_id: str, data_entry: Dict[str, Any]) -> None:
        """
        Anonymize data.
        
        Args:
            data_id: Data ID
            data_entry: Data entry
        """
        # Create anonymized data
        anonymized_data = {
            "id": data_id,
            "category": data_entry["category"],
            "timestamp": data_entry["timestamp"],
            "anonymized": True,
            "metadata": {
                "anonymized_at": time.time()
            }
        }
        
        # Replace data
        self.data_store[data_id] = anonymized_data
        logger.info(f"Anonymized data {data_id}")
    
    async def _pseudonymize_data(self, data_id: str, data_entry: Dict[str, Any]) -> None:
        """
        Pseudonymize data.
        
        Args:
            data_id: Data ID
            data_entry: Data entry
        """
        # Create pseudonymized data
        pseudonymized_data = {
            "id": f"pseudo_{data_id}",
            "category": data_entry["category"],
            "timestamp": data_entry["timestamp"],
            "pseudonymized": True,
            "metadata": {
                "pseudonymized_at": time.time()
            }
        }
        
        # Replace data
        self.data_store[data_id] = pseudonymized_data
        logger.info(f"Pseudonymized data {data_id}")
    
    def get_data_categories(self) -> List[DataCategory]:
        """
        Get all data categories.
        
        Returns:
            List of data categories
        """
        return list(self.policies.keys())
    
    def get_retention_periods(self) -> List[RetentionPeriod]:
        """
        Get all retention periods.
        
        Returns:
            List of retention periods
        """
        return list(RetentionPeriod)
    
    def get_policy_summary(self) -> Dict[str, Any]:
        """
        Get summary of retention policies.
        
        Returns:
            Dictionary with policy summary
        """
        summary = {}
        
        for category, policy in self.policies.items():
            summary[category.value] = {
                "retention_period": policy.retention_period.value,
                "retention_days": policy.retention_days,
                "purge_method": policy.purge_method,
                "requires_consent": policy.requires_consent,
                "legal_basis": policy.legal_basis,
                "description": policy.description
            }
        
        return summary
