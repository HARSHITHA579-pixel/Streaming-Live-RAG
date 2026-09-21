"""
Session State Management for Streaming RAG

================================================================================
RESPONSIBILITY:
- Maintain persistent, mutable conversational memory and evidence state for ONE session.
- Store the accumulated evidence pool, structured claims, citations, covered intents,
  and incremental answer versions.
- Provide query methods for the 2-Tier Controller to inspect covered intents and prior claims
  before making retrieval decisions.
- Support delta updates during session refinement WITHOUT clearing existing evidence or re-retrieving
  already satisfied intents.

INPUTS:
- Updates from `app.controller` (newly identified intents/sub-queries).
- Updates from `app.retriever` (new evidence chunks added to the pool).
- Updates from `app.synthesis` (new structured claims generated).
- Updates from `app.verifier` (verification statuses and grounded citation mappings).

OUTPUTS:
- Structured state snapshots for controller context inspection, synthesis prompting,
  and client streaming responses.

CONNECTED COMPONENTS:
- `app.controller`: Reads `covered_intents`, `previous_claims`, `previous_answer`, and `evidence_pool`
  prior to making the T1 decision.
- `app.retriever`: Appends new deduplicated chunks into `evidence_pool`.
- `app.synthesis`: Uses accumulated state to generate or update structured claims.
- `app.verifier`: Updates claim verification status and grounded citation list.
- `app.telemetry`: Emits state change snapshots into JSONL telemetry logs.

WHY THIS ARCHITECTURE:
- Live streaming conversations evolve incrementally (e.g. user says "International trip reimbursement
  is allowed", and later adds "The booking was made after the trip").
- Maintaining persistent state prevents redundant corpus retrieval and allows the system to focus
  exclusively on resolving delta intents while preserving previously verified facts.
================================================================================
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum


class ClaimStatus(str, Enum):
    """Status of a structured claim evaluated by the verifier."""
    UNVERIFIED = "UNVERIFIED"
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass
class RetrievedEvidence:
    """Represents a chunk retrieved from the corpus stored in the session evidence pool."""
    chunk_id: str
    doc_id: str
    section_id: Optional[str]
    section_title: Optional[str]
    text: str
    score: float
    sub_query_id: int
    retrieval_method: str  # "bm25", "dense", "rrf", "rerank"


@dataclass
class StructuredClaim:
    """
    Represents an atomic, citeable factual statement generated during synthesis.
    """
    id: int
    text: str
    cites: List[str] = field(default_factory=list)  # e.g. ["Doc_12 §2"]
    intent_id: int = 0
    status: ClaimStatus = ClaimStatus.UNVERIFIED
    verification_reasoning: Optional[str] = None
    softened_text: Optional[str] = None  # Populated if status is PARTIALLY_SUPPORTED


@dataclass
class SessionState:
    """
    Encapsulates all persistent state for a single live streaming session.
    """
    session_id: str
    answer_version: int = 0
    transcript_history: List[str] = field(default_factory=list)
    covered_intents: List[str] = field(default_factory=list)
    evidence_pool: Dict[str, RetrievedEvidence] = field(default_factory=dict)  # chunk_id -> RetrievedEvidence
    claims: List[StructuredClaim] = field(default_factory=list)
    citations: Set[str] = field(default_factory=set)
    previous_answer: Optional[str] = None
    uncertainty_notes: List[str] = field(default_factory=list)

    def get_covered_intents_summary(self) -> str:
        """
        Returns a formatted summary of already satisfied intents for the Controller context.
        """
        if not self.covered_intents:
            return "None"
        return ", ".join(self.covered_intents)

    def add_evidence(self, chunks: List[RetrievedEvidence]) -> None:
        """
        Merges new chunks into the evidence pool without duplicating existing chunk IDs.
        Preserves highest score when duplicates occur.
        """
        for chunk in chunks:
            if chunk.chunk_id not in self.evidence_pool:
                self.evidence_pool[chunk.chunk_id] = chunk
            else:
                # Update with higher score or richer metadata if available
                if chunk.score > self.evidence_pool[chunk.chunk_id].score:
                    self.evidence_pool[chunk.chunk_id] = chunk

    def update_claims(self, new_claims: List[StructuredClaim]) -> None:
        """
        Appends or updates structured claims in session state and registers unique citations.
        """
        claim_map = {c.id: c for c in self.claims}
        for claim in new_claims:
            claim_map[claim.id] = claim
            for cite in claim.cites:
                self.citations.add(cite)
        self.claims = list(claim_map.values())

    def record_refinement(self, new_transcript_chunk: str) -> None:
        """
        Increments answer_version and registers the incoming transcript chunk into history.
        """
        if new_transcript_chunk.strip():
            self.transcript_history.append(new_transcript_chunk.strip())
        self.answer_version += 1

    def reset(self) -> None:
        """
        Clears session state if a brand new conversation is explicitly started.
        """
        self.answer_version = 0
        self.transcript_history.clear()
        self.covered_intents.clear()
        self.evidence_pool.clear()
        self.claims.clear()
        self.citations.clear()
        self.previous_answer = None
        self.uncertainty_notes.clear()

