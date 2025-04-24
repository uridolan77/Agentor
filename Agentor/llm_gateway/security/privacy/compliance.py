"""
Compliance with privacy regulations for the LLM Gateway.

This module provides features for compliance with privacy regulations such as
GDPR, CCPA, and other data protection laws.
"""

import time
import logging
import json
from typing import Dict, List, Optional, Set, Union, Any, Tuple, Callable
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import uuid

from .pii import (
    PIIType,
    PIIDetection,
    PIIDetector,
    PIIRedactor
)
from .retention import (
    RetentionPeriod,
    DataCategory,
    RetentionPolicy,
    DataRetentionManager
)

logger = logging.getLogger(__name__)


class PrivacyRegulation(Enum):
    """Privacy regulations."""
    GDPR = "gdpr"  # General Data Protection Regulation (EU)
    CCPA = "ccpa"  # California Consumer Privacy Act (US)
    CPRA = "cpra"  # California Privacy Rights Act (US)
    LGPD = "lgpd"  # Lei Geral de Proteção de Dados (Brazil)
    PIPEDA = "pipeda"  # Personal Information Protection and Electronic Documents Act (Canada)
    POPIA = "popia"  # Protection of Personal Information Act (South Africa)
    PDPA = "pdpa"  # Personal Data Protection Act (Singapore)
    APP = "app"  # Australian Privacy Principles (Australia)
    OTHER = "other"  # Other privacy regulations


class ConsentType(Enum):
    """Types of user consent."""
    ESSENTIAL = "essential"  # Essential for service operation
    FUNCTIONAL = "functional"  # Functional cookies and features
    ANALYTICS = "analytics"  # Analytics and performance
    MARKETING = "marketing"  # Marketing and advertising
    THIRD_PARTY = "third_party"  # Third-party services


class ConsentRecord:
    """Record of user consent."""

    def __init__(
        self,
        user_id: str,
        consent_type: ConsentType,
        granted: bool,
        timestamp: Optional[float] = None,
        expiration: Optional[float] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize consent record.

        Args:
            user_id: User ID
            consent_type: Type of consent
            granted: Whether consent was granted
            timestamp: Timestamp of consent
            expiration: Expiration timestamp of consent
            ip_address: IP address of user
            user_agent: User agent of user
            metadata: Additional metadata
        """
        self.user_id = user_id
        self.consent_type = consent_type
        self.granted = granted
        self.timestamp = timestamp or time.time()
        self.expiration = expiration
        self.ip_address = ip_address
        self.user_agent = user_agent
        self.metadata = metadata or {}
        self.record_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert consent record to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "record_id": self.record_id,
            "user_id": self.user_id,
            "consent_type": self.consent_type.value,
            "granted": self.granted,
            "timestamp": self.timestamp,
            "expiration": self.expiration,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "metadata": self.metadata
        }

    def is_expired(self) -> bool:
        """
        Check if consent is expired.

        Returns:
            True if expired, False otherwise
        """
        if self.expiration is None:
            return False

        return time.time() > self.expiration


class DataSubjectRequest(Enum):
    """Types of data subject requests."""
    ACCESS = "access"  # Right to access personal data
    RECTIFICATION = "rectification"  # Right to rectify inaccurate personal data
    ERASURE = "erasure"  # Right to erasure ("right to be forgotten")
    RESTRICTION = "restriction"  # Right to restrict processing
    PORTABILITY = "portability"  # Right to data portability
    OBJECTION = "objection"  # Right to object to processing
    AUTOMATED_DECISION = "automated_decision"  # Rights related to automated decision making
    CONSENT_WITHDRAWAL = "consent_withdrawal"  # Right to withdraw consent


class DataSubjectRequestStatus(Enum):
    """Status of data subject requests."""
    PENDING = "pending"  # Request is pending
    IN_PROGRESS = "in_progress"  # Request is being processed
    COMPLETED = "completed"  # Request has been completed
    DENIED = "denied"  # Request has been denied
    CANCELLED = "cancelled"  # Request has been cancelled


class DataSubjectRequestRecord:
    """Record of a data subject request."""

    def __init__(
        self,
        user_id: str,
        request_type: DataSubjectRequest,
        status: DataSubjectRequestStatus = DataSubjectRequestStatus.PENDING,
        timestamp: Optional[float] = None,
        completion_timestamp: Optional[float] = None,
        request_data: Optional[Dict[str, Any]] = None,
        response_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize data subject request record.

        Args:
            user_id: User ID
            request_type: Type of request
            status: Status of request
            timestamp: Timestamp of request
            completion_timestamp: Timestamp of request completion
            request_data: Request data
            response_data: Response data
            metadata: Additional metadata
        """
        self.user_id = user_id
        self.request_type = request_type
        self.status = status
        self.timestamp = timestamp or time.time()
        self.completion_timestamp = completion_timestamp
        self.request_data = request_data or {}
        self.response_data = response_data or {}
        self.metadata = metadata or {}
        self.request_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert request record to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "request_type": self.request_type.value,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "completion_timestamp": self.completion_timestamp,
            "request_data": self.request_data,
            "response_data": self.response_data,
            "metadata": self.metadata
        }

    def update_status(
        self,
        status: DataSubjectRequestStatus,
        response_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update request status.

        Args:
            status: New status
            response_data: Response data
        """
        self.status = status

        if response_data:
            self.response_data.update(response_data)

        if status == DataSubjectRequestStatus.COMPLETED:
            self.completion_timestamp = time.time()


class PrivacyNotice:
    """Privacy notice for users."""

    def __init__(
        self,
        version: str,
        effective_date: float,
        content: str,
        summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize privacy notice.

        Args:
            version: Version of the notice
            effective_date: Effective date of the notice
            content: Content of the notice
            summary: Summary of the notice
            metadata: Additional metadata
        """
        self.version = version
        self.effective_date = effective_date
        self.content = content
        self.summary = summary
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert privacy notice to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "version": self.version,
            "effective_date": self.effective_date,
            "content": self.content,
            "summary": self.summary,
            "metadata": self.metadata
        }

    def is_current(self) -> bool:
        """
        Check if privacy notice is current.

        Returns:
            True if current, False otherwise
        """
        return self.effective_date <= time.time()


class PrivacyNoticeAcceptance:
    """Record of user acceptance of a privacy notice."""

    def __init__(
        self,
        user_id: str,
        notice_version: str,
        accepted: bool,
        timestamp: Optional[float] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize privacy notice acceptance.

        Args:
            user_id: User ID
            notice_version: Version of the notice
            accepted: Whether the notice was accepted
            timestamp: Timestamp of acceptance
            ip_address: IP address of user
            user_agent: User agent of user
            metadata: Additional metadata
        """
        self.user_id = user_id
        self.notice_version = notice_version
        self.accepted = accepted
        self.timestamp = timestamp or time.time()
        self.ip_address = ip_address
        self.user_agent = user_agent
        self.metadata = metadata or {}
        self.acceptance_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert acceptance record to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "acceptance_id": self.acceptance_id,
            "user_id": self.user_id,
            "notice_version": self.notice_version,
            "accepted": self.accepted,
            "timestamp": self.timestamp,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "metadata": self.metadata
        }


class ComplianceManager:
    """Manager for privacy compliance."""

    def __init__(
        self,
        pii_detector: Optional[PIIDetector] = None,
        pii_redactor: Optional[PIIRedactor] = None,
        retention_manager: Optional[DataRetentionManager] = None,
        regulations: Optional[Set[PrivacyRegulation]] = None,
        consent_records: Optional[Dict[str, List[ConsentRecord]]] = None,
        dsr_records: Optional[Dict[str, List[DataSubjectRequestRecord]]] = None,
        privacy_notices: Optional[Dict[str, PrivacyNotice]] = None,
        notice_acceptances: Optional[Dict[str, List[PrivacyNoticeAcceptance]]] = None
    ):
        """
        Initialize compliance manager.

        Args:
            pii_detector: PII detector
            pii_redactor: PII redactor
            retention_manager: Data retention manager
            regulations: Set of privacy regulations to comply with
            consent_records: Dictionary mapping user IDs to consent records
            dsr_records: Dictionary mapping user IDs to data subject request records
            privacy_notices: Dictionary mapping versions to privacy notices
            notice_acceptances: Dictionary mapping user IDs to privacy notice acceptances
        """
        self.pii_detector = pii_detector or PIIDetector()
        self.pii_redactor = pii_redactor or PIIRedactor(detector=self.pii_detector)
        self.retention_manager = retention_manager or DataRetentionManager()
        self.regulations = regulations or {PrivacyRegulation.GDPR, PrivacyRegulation.CCPA}

        # Storage for compliance records
        self.consent_records = consent_records or {}
        self.dsr_records = dsr_records or {}
        self.privacy_notices = privacy_notices or {}
        self.notice_acceptances = notice_acceptances or {}

        # Lock for thread safety
        self.lock = asyncio.Lock()

    async def detect_and_redact_pii(self, text: str) -> Tuple[str, List[PIIDetection]]:
        """
        Detect and redact PII in text.

        Args:
            text: Text to process

        Returns:
            Tuple of (redacted text, list of detections)
        """
        return self.pii_redactor.redact(text)

    async def record_consent(
        self,
        user_id: str,
        consent_type: ConsentType,
        granted: bool,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConsentRecord:
        """
        Record user consent.

        Args:
            user_id: User ID
            consent_type: Type of consent
            granted: Whether consent was granted
            ip_address: IP address of user
            user_agent: User agent of user
            metadata: Additional metadata

        Returns:
            Consent record
        """
        # Create consent record
        record = ConsentRecord(
            user_id=user_id,
            consent_type=consent_type,
            granted=granted,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata=metadata
        )

        # Store record
        async with self.lock:
            if user_id not in self.consent_records:
                self.consent_records[user_id] = []

            self.consent_records[user_id].append(record)

        return record

    async def get_user_consent(
        self,
        user_id: str,
        consent_type: Optional[ConsentType] = None
    ) -> List[ConsentRecord]:
        """
        Get user consent records.

        Args:
            user_id: User ID
            consent_type: Type of consent to filter by

        Returns:
            List of consent records
        """
        async with self.lock:
            # Get all consent records for user
            records = self.consent_records.get(user_id, [])

            # Filter by consent type if specified
            if consent_type:
                records = [r for r in records if r.consent_type == consent_type]

            # Sort by timestamp (newest first)
            records.sort(key=lambda r: r.timestamp, reverse=True)

            return records

    async def has_user_consent(
        self,
        user_id: str,
        consent_type: ConsentType
    ) -> bool:
        """
        Check if user has given consent.

        Args:
            user_id: User ID
            consent_type: Type of consent

        Returns:
            True if user has given consent, False otherwise
        """
        # Get consent records for user and type
        records = await self.get_user_consent(user_id, consent_type)

        # Check if any valid consent records exist
        for record in records:
            if record.granted and not record.is_expired():
                return True

        return False

    async def create_data_subject_request(
        self,
        user_id: str,
        request_type: DataSubjectRequest,
        request_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DataSubjectRequestRecord:
        """
        Create a data subject request.

        Args:
            user_id: User ID
            request_type: Type of request
            request_data: Request data
            metadata: Additional metadata

        Returns:
            Data subject request record
        """
        # Create request record
        record = DataSubjectRequestRecord(
            user_id=user_id,
            request_type=request_type,
            request_data=request_data,
            metadata=metadata
        )

        # Store record
        async with self.lock:
            if user_id not in self.dsr_records:
                self.dsr_records[user_id] = []

            self.dsr_records[user_id].append(record)

        return record

    async def get_data_subject_requests(
        self,
        user_id: Optional[str] = None,
        request_type: Optional[DataSubjectRequest] = None,
        status: Optional[DataSubjectRequestStatus] = None
    ) -> List[DataSubjectRequestRecord]:
        """
        Get data subject request records.

        Args:
            user_id: User ID to filter by
            request_type: Type of request to filter by
            status: Status of request to filter by

        Returns:
            List of data subject request records
        """
        async with self.lock:
            # Get all records
            all_records = []

            if user_id:
                # Get records for specific user
                all_records.extend(self.dsr_records.get(user_id, []))
            else:
                # Get records for all users
                for user_records in self.dsr_records.values():
                    all_records.extend(user_records)

            # Filter by request type if specified
            if request_type:
                all_records = [r for r in all_records if r.request_type == request_type]

            # Filter by status if specified
            if status:
                all_records = [r for r in all_records if r.status == status]

            # Sort by timestamp (newest first)
            all_records.sort(key=lambda r: r.timestamp, reverse=True)

            return all_records

    async def update_data_subject_request(
        self,
        request_id: str,
        status: DataSubjectRequestStatus,
        response_data: Optional[Dict[str, Any]] = None
    ) -> Optional[DataSubjectRequestRecord]:
        """
        Update a data subject request.

        Args:
            request_id: Request ID
            status: New status
            response_data: Response data

        Returns:
            Updated request record or None if not found
        """
        async with self.lock:
            # Find request record
            for user_id, user_records in self.dsr_records.items():
                for record in user_records:
                    if record.request_id == request_id:
                        # Update record
                        record.update_status(status, response_data)
                        return record

        return None

    async def add_privacy_notice(
        self,
        version: str,
        content: str,
        effective_date: Optional[float] = None,
        summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PrivacyNotice:
        """
        Add a privacy notice.

        Args:
            version: Version of the notice
            content: Content of the notice
            effective_date: Effective date of the notice
            summary: Summary of the notice
            metadata: Additional metadata

        Returns:
            Privacy notice
        """
        # Create privacy notice
        notice = PrivacyNotice(
            version=version,
            effective_date=effective_date or time.time(),
            content=content,
            summary=summary,
            metadata=metadata
        )

        # Store notice
        async with self.lock:
            self.privacy_notices[version] = notice

        return notice

    async def get_privacy_notice(
        self,
        version: Optional[str] = None
    ) -> Optional[PrivacyNotice]:
        """
        Get a privacy notice.

        Args:
            version: Version of the notice

        Returns:
            Privacy notice or None if not found
        """
        async with self.lock:
            if version:
                # Get specific version
                return self.privacy_notices.get(version)
            else:
                # Get latest version
                if not self.privacy_notices:
                    return None

                # Sort by effective date (newest first)
                notices = list(self.privacy_notices.values())
                notices.sort(key=lambda n: n.effective_date, reverse=True)

                return notices[0]

    async def record_privacy_notice_acceptance(
        self,
        user_id: str,
        notice_version: str,
        accepted: bool,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PrivacyNoticeAcceptance:
        """
        Record user acceptance of a privacy notice.

        Args:
            user_id: User ID
            notice_version: Version of the notice
            accepted: Whether the notice was accepted
            ip_address: IP address of user
            user_agent: User agent of user
            metadata: Additional metadata

        Returns:
            Privacy notice acceptance record
        """
        # Create acceptance record
        record = PrivacyNoticeAcceptance(
            user_id=user_id,
            notice_version=notice_version,
            accepted=accepted,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata=metadata
        )

        # Store record
        async with self.lock:
            if user_id not in self.notice_acceptances:
                self.notice_acceptances[user_id] = []

            self.notice_acceptances[user_id].append(record)

        return record

    async def has_accepted_privacy_notice(
        self,
        user_id: str,
        notice_version: Optional[str] = None
    ) -> bool:
        """
        Check if user has accepted a privacy notice.

        Args:
            user_id: User ID
            notice_version: Version of the notice

        Returns:
            True if user has accepted the notice, False otherwise
        """
        async with self.lock:
            # Get all acceptance records for user
            records = self.notice_acceptances.get(user_id, [])

            if not records:
                return False

            if notice_version:
                # Check specific version
                for record in records:
                    if record.notice_version == notice_version and record.accepted:
                        return True

                return False
            else:
                # Get latest notice version
                latest_notice = await self.get_privacy_notice()
                if not latest_notice:
                    return False

                # Check if user has accepted latest version
                for record in records:
                    if record.notice_version == latest_notice.version and record.accepted:
                        return True

                return False

    async def process_data_subject_request(
        self,
        request_id: str
    ) -> Optional[DataSubjectRequestRecord]:
        """
        Process a data subject request.

        Args:
            request_id: Request ID

        Returns:
            Updated request record or None if not found
        """
        # Find request record
        request_record = None
        async with self.lock:
            for user_id, user_records in self.dsr_records.items():
                for record in user_records:
                    if record.request_id == request_id:
                        request_record = record
                        break
                if request_record:
                    break

        if not request_record:
            logger.warning(f"Data subject request {request_id} not found")
            return None

        # Update status to in progress
        await self.update_data_subject_request(
            request_id=request_id,
            status=DataSubjectRequestStatus.IN_PROGRESS
        )

        # Process request based on type
        user_id = request_record.user_id
        request_type = request_record.request_type

        try:
            if request_type == DataSubjectRequest.ACCESS:
                # Gather user data
                user_data = await self._gather_user_data(user_id)

                # Update request with data
                await self.update_data_subject_request(
                    request_id=request_id,
                    status=DataSubjectRequestStatus.COMPLETED,
                    response_data={"user_data": user_data}
                )

            elif request_type == DataSubjectRequest.ERASURE:
                # Delete user data
                await self._delete_user_data(user_id)

                # Update request
                await self.update_data_subject_request(
                    request_id=request_id,
                    status=DataSubjectRequestStatus.COMPLETED,
                    response_data={"deleted": True}
                )

            elif request_type == DataSubjectRequest.PORTABILITY:
                # Gather user data in portable format
                user_data = await self._gather_user_data(user_id)

                # Convert to portable format (JSON)
                portable_data = json.dumps(user_data, indent=2)

                # Update request with data
                await self.update_data_subject_request(
                    request_id=request_id,
                    status=DataSubjectRequestStatus.COMPLETED,
                    response_data={"portable_data": portable_data}
                )

            elif request_type == DataSubjectRequest.CONSENT_WITHDRAWAL:
                # Withdraw all consent
                await self._withdraw_user_consent(user_id)

                # Update request
                await self.update_data_subject_request(
                    request_id=request_id,
                    status=DataSubjectRequestStatus.COMPLETED,
                    response_data={"consent_withdrawn": True}
                )

            else:
                # Not implemented yet
                await self.update_data_subject_request(
                    request_id=request_id,
                    status=DataSubjectRequestStatus.DENIED,
                    response_data={"reason": f"Request type {request_type.value} not implemented yet"}
                )

        except Exception as e:
            logger.error(f"Error processing data subject request {request_id}: {e}")

            # Update request with error
            await self.update_data_subject_request(
                request_id=request_id,
                status=DataSubjectRequestStatus.DENIED,
                response_data={"error": str(e)}
            )

        # Get updated request record
        async with self.lock:
            for user_id, user_records in self.dsr_records.items():
                for record in user_records:
                    if record.request_id == request_id:
                        return record

        return None

    async def _gather_user_data(self, user_id: str) -> Dict[str, Any]:
        """
        Gather all data for a user.

        Args:
            user_id: User ID

        Returns:
            Dictionary with user data
        """
        user_data = {
            "user_id": user_id,
            "consent_records": [],
            "privacy_notice_acceptances": [],
            "data_subject_requests": []
        }

        async with self.lock:
            # Get consent records
            for record in self.consent_records.get(user_id, []):
                user_data["consent_records"].append(record.to_dict())

            # Get privacy notice acceptances
            for record in self.notice_acceptances.get(user_id, []):
                user_data["privacy_notice_acceptances"].append(record.to_dict())

            # Get data subject requests
            for record in self.dsr_records.get(user_id, []):
                user_data["data_subject_requests"].append(record.to_dict())

        return user_data

    async def _delete_user_data(self, user_id: str) -> None:
        """
        Delete all data for a user.

        Args:
            user_id: User ID
        """
        async with self.lock:
            # Delete consent records
            if user_id in self.consent_records:
                del self.consent_records[user_id]

            # Delete privacy notice acceptances
            if user_id in self.notice_acceptances:
                del self.notice_acceptances[user_id]

            # Keep data subject requests for compliance
            # But anonymize them
            if user_id in self.dsr_records:
                for record in self.dsr_records[user_id]:
                    record.user_id = f"deleted_{user_id}"
                    record.metadata["anonymized"] = True

        # Delete data from retention manager
        if self.retention_manager:
            # This is a simplified example
            # In a real implementation, you would delete all user data
            # from all data stores
            pass

    async def _withdraw_user_consent(self, user_id: str) -> None:
        """
        Withdraw all consent for a user.

        Args:
            user_id: User ID
        """
        async with self.lock:
            # Add withdrawal records for all consent types
            for consent_type in ConsentType:
                await self.record_consent(
                    user_id=user_id,
                    consent_type=consent_type,
                    granted=False,
                    metadata={"withdrawal": True}
                )

    def get_supported_regulations(self) -> Set[PrivacyRegulation]:
        """
        Get supported privacy regulations.

        Returns:
            Set of supported regulations
        """
        return self.regulations

    def add_regulation(self, regulation: PrivacyRegulation) -> None:
        """
        Add a supported privacy regulation.

        Args:
            regulation: Privacy regulation
        """
        self.regulations.add(regulation)

    def remove_regulation(self, regulation: PrivacyRegulation) -> None:
        """
        Remove a supported privacy regulation.

        Args:
            regulation: Privacy regulation
        """
        if regulation in self.regulations:
            self.regulations.remove(regulation)

    def get_compliance_summary(self) -> Dict[str, Any]:
        """
        Get summary of compliance status.

        Returns:
            Dictionary with compliance summary
        """
        summary = {
            "regulations": [r.value for r in self.regulations],
            "privacy_notices": len(self.privacy_notices),
            "users_with_consent": len(self.consent_records),
            "data_subject_requests": {
                "total": sum(len(records) for records in self.dsr_records.values()),
                "pending": 0,
                "in_progress": 0,
                "completed": 0,
                "denied": 0,
                "cancelled": 0
            }
        }

        # Count data subject requests by status
        for user_records in self.dsr_records.values():
            for record in user_records:
                summary["data_subject_requests"][record.status.value] += 1

        return summary
