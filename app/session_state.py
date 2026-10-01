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

import re
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
    page_number: int = 1
    geographic_scope: str = "General"
    scope_level: str = "general"
    source_priority: int = 4
    source_file: str = ""
    source_path: str = ""
    document_title: Optional[str] = None



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
    Distinguishes active context vs historical/superseded context.
    """
    session_id: str
    answer_version: int = 0
    transcript_history: List[str] = field(default_factory=list)
    covered_intents: List[str] = field(default_factory=list)
    active_intents: List[str] = field(default_factory=list)
    superseded_intents: List[str] = field(default_factory=list)
    evidence_pool: Dict[str, RetrievedEvidence] = field(default_factory=dict)  # All historical evidence
    active_evidence_pool: Dict[str, RetrievedEvidence] = field(default_factory=dict)  # Active turn evidence
    superseded_evidence_pool: Dict[str, RetrievedEvidence] = field(default_factory=dict)  # Stale/superseded
    claims: List[StructuredClaim] = field(default_factory=list)
    active_claims: List[StructuredClaim] = field(default_factory=list)
    citations: Set[str] = field(default_factory=set)
    previous_answer: Optional[str] = None
    uncertainty_notes: List[str] = field(default_factory=list)

    def get_covered_intents_summary(self) -> str:
        """
        Returns a formatted summary of already satisfied intents for the Controller context.
        """
        if not self.active_intents and not self.covered_intents:
            return "None"
        return ", ".join(self.active_intents or self.covered_intents)

    def supersede_active_context(self, new_intent: Optional[str] = None) -> None:
        """
        Marks all current active evidence and intents as superseded/stale when
        a user replacement or correction occurs.
        """
        for k, v in self.active_evidence_pool.items():
            self.superseded_evidence_pool[k] = v
        self.active_evidence_pool.clear()

        for intent in self.active_intents:
            if intent not in self.superseded_intents:
                self.superseded_intents.append(intent)
        self.active_intents.clear()
        if new_intent and new_intent.strip():
            self.active_intents.append(new_intent.strip())

        self.active_claims.clear()
        self.citations.clear()

    def remove_intent_context(self, removed_topic: str) -> None:
        """
        Explicitly removes a specific topic from active intents and moves
        associated evidence chunks to superseded pool.
        """
        clean_topic = removed_topic.strip().lower()
        if not clean_topic:
            return

        topic_words = set(re.findall(r"\w+", clean_topic))
        # Move matching active chunks to superseded pool
        remaining_active = {}
        for chunk_id, chunk in self.active_evidence_pool.items():
            chunk_words = set(re.findall(r"\w+", chunk.text.lower()))
            if topic_words.intersection(chunk_words):
                self.superseded_evidence_pool[chunk_id] = chunk
            else:
                remaining_active[chunk_id] = chunk
        self.active_evidence_pool = remaining_active

        # Update active intents
        new_active_intents = []
        for intent in self.active_intents:
            if any(w in intent.lower() for w in topic_words):
                self.superseded_intents.append(intent)
            else:
                new_active_intents.append(intent)
        self.active_intents = new_active_intents

        # Filter active claims
        self.active_claims = [c for c in self.active_claims if not any(w in c.text.lower() for w in topic_words)]
        self.citations = {cite for c in self.active_claims for cite in c.cites}

    def add_evidence(self, chunks: List[RetrievedEvidence], is_supersession: bool = False) -> None:
        """
        Merges new chunks into the active evidence pool and historical evidence pool.
        If is_supersession is True, replaces the active pool with the newly retrieved chunks.
        """
        if is_supersession:
            self.supersede_active_context()

        for chunk in chunks:
            # Always update historical pool
            if chunk.chunk_id not in self.evidence_pool or chunk.score > self.evidence_pool[chunk.chunk_id].score:
                self.evidence_pool[chunk.chunk_id] = chunk

            # Update active pool
            if chunk.chunk_id not in self.active_evidence_pool or chunk.score > self.active_evidence_pool[chunk.chunk_id].score:
                self.active_evidence_pool[chunk.chunk_id] = chunk

    def set_active_claims(self, new_claims: List[StructuredClaim]) -> None:
        """
        Sets the active claims for the current answer version and updates citations.
        """
        self.active_claims = list(new_claims)
        self.claims.extend([c for c in new_claims if c not in self.claims])
        self.citations = {cite for c in new_claims for cite in c.cites}

    def update_claims(self, new_claims: List[StructuredClaim]) -> None:
        """
        Appends or updates structured claims in session state and registers unique citations.
        """
        self.set_active_claims(new_claims)

    def record_refinement(self, new_transcript_chunk: str, is_supersession: bool = False) -> None:
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
        self.active_intents.clear()
        self.superseded_intents.clear()
        self.evidence_pool.clear()
        self.active_evidence_pool.clear()
        self.superseded_evidence_pool.clear()
        self.claims.clear()
        self.active_claims.clear()
        self.citations.clear()
        self.previous_answer = None
        self.uncertainty_notes.clear()

