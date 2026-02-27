"""Verification system for HolonPolis.

Validates code changes: what changed, whether it took effect, where is the evidence.
"""

from holonpolis.kernel.verification.evidence_collector import (
    EvidenceType,
    FileEvidence,
    ToolEvidence,
    VerificationEvidence,
    LLMEvidence,
    EvidencePackage,
    EvidenceCollector,
    create_evidence_collector,
)
from holonpolis.kernel.verification.change_verifier import (
    ChangeVerifier,
    VerificationResult,
    verify_file_changes,
    verify_syntax,
)

__all__ = [
    # Evidence types
    "EvidenceType",
    "FileEvidence",
    "ToolEvidence",
    "VerificationEvidence",
    "LLMEvidence",
    "EvidencePackage",
    "EvidenceCollector",
    "create_evidence_collector",
    # Change verification
    "ChangeVerifier",
    "VerificationResult",
    "verify_file_changes",
    "verify_syntax",
]
