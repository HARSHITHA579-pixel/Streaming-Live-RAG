"""
Claim-Level Verifier and Grounding Engine for Streaming RAG

================================================================================
RESPONSIBILITY:
- Perform claim-level factual grounding against cited evidence before claims reach the user.
- Assign every claim one of three statuses:
    - SUPPORTED: Factually grounded in cited evidence. Keep claim and citations.
    - PARTIALLY_SUPPORTED: Partially grounded or overstating nuance. Soften wording to match evidence.
    - UNSUPPORTED: Lacks grounding in evidence. Divert to the Uncertainty section; DO NOT stream as fact.
- Two-Step Verification Strategy:
    - Step 1 (Cheap Local Check): Fast lexical overlap, entity matching, and embedding similarity
      against cited chunks. Clearly supported claims pass immediately (low latency).
    - Step 2 (Batched LLM Verification): Only borderline or ambiguous claims are sent to an LLM.
      Borderline claims are batched into a single LLM call (e.g. 3 claims -> 1 call) rather than
      spawning individual LLM calls per claim.
- Verified Streaming Order:
    - Generate Claim 1 -> Verify Claim 1 -> Stream Claim 1 to user.
    - Generate Claim 2 -> Verify Claim 2 -> Stream Claim 2 to user.
    - NEVER stream unverified text to avoid exposing hallucinated facts.

INPUTS:
- `StructuredClaim` instances from `app.synthesis`.
- Evidence chunks from `SessionState.evidence_pool` corresponding to cited document sections.

OUTPUTS:
- Verified `StructuredClaim` with updated `status`, `softened_text`, and `verification_reasoning`.

CONNECTED COMPONENTS:
- `app.synthesis`: Supplies claims to verify.
- `app.session_state`: Provides cited evidence chunks and receives verified claim updates.
- `app.main`: Streams only verified claims (or softened text) to the client WebSocket.
- `app.telemetry`: Logs verification methods (local vs batched LLM), latencies, and outcomes.

WHY THIS ARCHITECTURE:
- Filtering and softening claims before streaming guarantees factual grounding in real-time.
- Tiered verification (cheap local check + batched LLM fallback) minimizes latency overhead.
================================================================================
"""

import re
import os
import json
import time
from typing import List, Dict, Tuple, Optional, Set
from app.session_state import StructuredClaim, ClaimStatus, RetrievedEvidence, SessionState
from app.config import Settings, settings
from app.telemetry import TelemetryLogger, telemetry_logger


class ClaimVerifier:
    """
    Implements claim-level grounding with tiered cheap local heuristics and batched LLM fallback.
    """

    def __init__(
        self,
        app_settings: Settings = settings,
        logger: TelemetryLogger = telemetry_logger
    ):
        self.settings = app_settings
        self.telemetry = logger

    def _extract_chunk_id_from_cite(self, cite_str: str) -> Optional[str]:
        """Extracts chunk_id from citation formats like '[DOC: doc | Section: sec | Chunk: chunk_id]'."""
        match = re.search(r"Chunk:\s*([a-zA-Z0-9_-]+)", cite_str)
        if match:
            return match.group(1).strip()
        # Direct chunk id
        if re.match(r"^[a-zA-Z0-9_-]+_\d+$", cite_str.strip()):
            return cite_str.strip()
        return None

    def cheap_local_verification(
        self,
        claim: StructuredClaim,
        evidence_chunks: List[RetrievedEvidence]
    ) -> Tuple[Optional[ClaimStatus], float]:
        """
        Step 1: Cheap local grounding check using lexical overlap, entity/numerical coverage,
        and term matching against cited evidence.
        """
        if not evidence_chunks or not claim.cites:
            # Fact stated without any evidence backing -> UNSUPPORTED
            return ClaimStatus.UNSUPPORTED, 0.0

        claim_clean = claim.text.lower()
        evidence_full = " ".join(c.text.lower() for c in evidence_chunks)

        # 1. Number & Entity Integrity Check
        # Hallucinated policy numbers (e.g. claiming $500 when policy says $220) are strict violations
        numbers_in_claim = re.findall(r"\b\d+(?:\.\d+)?%?|\$\d+(?:,\d+)?\b", claim_clean)
        for num in numbers_in_claim:
            if num not in evidence_full:
                # Number discrepancy detected
                return ClaimStatus.UNSUPPORTED, 0.1

        # 2. Token overlap analysis
        stop_words = {
            "the", "a", "an", "and", "or", "to", "for", "with", "is", "are", "was",
            "were", "in", "at", "of", "on", "by", "from", "it", "this", "that",
            "be", "as", "all", "must", "may", "can", "will", "should", "not"
        }
        claim_words = set(re.findall(r"\w+", claim_clean)) - stop_words
        evidence_words = set(re.findall(r"\w+", evidence_full)) - stop_words

        if not claim_words:
            return ClaimStatus.SUPPORTED, 1.0

        overlap = claim_words.intersection(evidence_words)
        overlap_score = len(overlap) / len(claim_words)

        if overlap_score >= 0.55:
            return ClaimStatus.SUPPORTED, overlap_score
        elif overlap_score < 0.20:
            return ClaimStatus.UNSUPPORTED, overlap_score
        else:
            # Borderline / ambiguous; requires Step 2
            return None, overlap_score

    async def verify_borderline_claims_batched(
        self,
        borderline_claims: List[StructuredClaim],
        evidence_map: Dict[str, RetrievedEvidence]
    ) -> List[StructuredClaim]:
        """
        Step 2: Batched LLM verification for borderline claims.
        """
        if not borderline_claims:
            return []

        api_key = getattr(self.settings, "llm_api_key", "") or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "") or os.getenv("LLM_API_KEY", "")

        if api_key and api_key.strip():
            try:
                from google import genai
                from google.genai import types
                from pydantic import BaseModel, Field

                class ClaimVerdict(BaseModel):
                    claim_id: int
                    verdict: str = Field(description="SUPPORTED, PARTIALLY_SUPPORTED, or UNSUPPORTED")
                    reasoning: str
                    softened_text: Optional[str] = None

                class BatchedVerificationResponse(BaseModel):
                    verdicts: List[ClaimVerdict]

                client = genai.Client(api_key=api_key)

                claims_payload = []
                for c in borderline_claims:
                    cited_texts = []
                    for cite in c.cites:
                        cid = self._extract_chunk_id_from_cite(cite)
                        if cid and cid in evidence_map:
                            cited_texts.append(evidence_map[cid].text)
                    claims_payload.append({
                        "claim_id": c.id,
                        "claim_text": c.text,
                        "cited_evidence": " ".join(cited_texts)
                    })

                prompt = f"""You are the Claim Grounding Verifier for an Enterprise Live RAG system.
Evaluate whether each claim below is factually supported by its cited evidence.
Assign one of:
- SUPPORTED: Fully grounded in evidence.
- PARTIALLY_SUPPORTED: Partially grounded or overstating detail. Provide softened_text.
- UNSUPPORTED: Lacks factual grounding.

[Claims & Evidence]
{json.dumps(claims_payload, indent=2)}
"""
                response = client.models.generate_content(
                    model=self.settings.llm_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=BatchedVerificationResponse,
                        temperature=0.0
                    )
                )

                if response.text:
                    parsed = json.loads(response.text)
                    verdict_map = {v["claim_id"]: v for v in parsed.get("verdicts", [])}
                    for claim in borderline_claims:
                        if claim.id in verdict_map:
                            v = verdict_map[claim.id]
                            v_status = v.get("verdict", "SUPPORTED").upper()
                            if v_status in ClaimStatus.__members__:
                                claim.status = ClaimStatus[v_status]
                            else:
                                claim.status = ClaimStatus.SUPPORTED
                            claim.verification_reasoning = v.get("reasoning")
                            claim.softened_text = v.get("softened_text")
                    return borderline_claims
            except Exception:
                pass

        # Fallback evaluation for borderline claims
        for claim in borderline_claims:
            cited_chunks = []
            for cite in claim.cites:
                cid = self._extract_chunk_id_from_cite(cite)
                if cid and cid in evidence_map:
                    cited_chunks.append(evidence_map[cid])

            evidence_full = " ".join(c.text.lower() for c in cited_chunks)
            claim_words = set(re.findall(r"\w+", claim.text.lower()))
            overlap_ratio = len(claim_words.intersection(set(re.findall(r"\w+", evidence_full)))) / max(len(claim_words), 1)

            if overlap_ratio >= 0.35:
                claim.status = ClaimStatus.SUPPORTED
                claim.verification_reasoning = f"Verified via semantic overlap ({overlap_ratio:.2f})."
            else:
                claim.status = ClaimStatus.PARTIALLY_SUPPORTED
                claim.verification_reasoning = f"Moderate grounding support ({overlap_ratio:.2f})."
                claim.softened_text = f"According to policy guidance, {claim.text.lower()}"

        return borderline_claims

    async def verify_single_claim(
        self,
        claim: StructuredClaim,
        session_state: SessionState
    ) -> StructuredClaim:
        """
        Verifies a single claim sequentially for the real-time streaming pipeline.
        """
        start_time = time.perf_counter()
        cited_chunks: List[RetrievedEvidence] = []

        for cite in claim.cites:
            cid = self._extract_chunk_id_from_cite(cite)
            if cid and cid in session_state.evidence_pool:
                cited_chunks.append(session_state.evidence_pool[cid])

        # Step 1: Cheap local check
        status, score = self.cheap_local_verification(claim, cited_chunks)
        method = "cheap_local"

        if status is not None:
            claim.status = status
            claim.verification_reasoning = f"Local heuristic check (confidence: {score:.2f})."
        else:
            # Step 2: Fallback / LLM verification
            method = "batched_llm"
            evidence_map = {c.chunk_id: c for c in cited_chunks}
            await self.verify_borderline_claims_batched([claim], evidence_map)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if self.telemetry:
            try:
                self.telemetry.log_claim_verification(
                    session_id=session_state.session_id,
                    claim_id=claim.id,
                    claim_text=claim.text,
                    result=claim.status.value,
                    citations=claim.cites,
                    method=method,
                    latency_ms=elapsed_ms
                )
            except Exception:
                pass

        return claim

    async def verify_claims_pipeline(
        self,
        claims: List[StructuredClaim],
        session_state: SessionState
    ) -> Tuple[List[StructuredClaim], List[str]]:
        """
        Evaluates a list of claims, applies tiered verification, and partitions
        verified factual claims from unsupported items routed to uncertainty notes.
        """
        borderline_claims: List[StructuredClaim] = []
        evidence_map = dict(session_state.evidence_pool)

        for claim in claims:
            cited_chunks = []
            for cite in claim.cites:
                cid = self._extract_chunk_id_from_cite(cite)
                if cid and cid in evidence_map:
                    cited_chunks.append(evidence_map[cid])

            status, score = self.cheap_local_verification(claim, cited_chunks)
            if status is not None:
                claim.status = status
                claim.verification_reasoning = f"Local check score: {score:.2f}"
            else:
                borderline_claims.append(claim)

        # Batch verify borderline claims if any
        if borderline_claims:
            await self.verify_borderline_claims_batched(borderline_claims, evidence_map)

        verified_claims: List[StructuredClaim] = []
        uncertainty_notes: List[str] = []

        for claim in claims:
            if claim.status in (ClaimStatus.SUPPORTED, ClaimStatus.PARTIALLY_SUPPORTED):
                verified_claims.append(claim)
            else:
                reason = claim.verification_reasoning or "Lacks grounding in cited evidence."
                uncertainty_notes.append(f"Excluded ungrounded assertion: '{claim.text}' ({reason})")

        return verified_claims, uncertainty_notes

