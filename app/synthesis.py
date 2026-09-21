"""
Structured Claim Synthesis for Streaming RAG

================================================================================
RESPONSIBILITY:
- Generate atomic, structured factual claims from retrieved evidence or existing session state,
  instead of generating unstructured free-form text directly.
- Each claim MUST specify:
    - id (int)
    - text (str): The factual assertion
    - cites (List[str]): Explicit section-level citations (e.g., ["Doc_12 §2"])
    - intent (int): Associated sub-query / intent index
- Support incremental generation: Yield claims sequentially so each claim can be verified
  immediately by the Verifier before being streamed to the user.
- Render verified structured claims into coherent, human-readable streaming responses with citations.
- Support no-retrieval synthesis (e.g. re-formatting previous answer based on SessionState).

INPUTS:
- Current transcript / user query.
- Retrieved evidence set from `app.retriever` or existing `SessionState.evidence_pool`.
- Prior claims and answer history from `SessionState`.

OUTPUTS:
- Generator / Stream of `StructuredClaim` objects for downstream verification and rendering.

CONNECTED COMPONENTS:
- `app.retriever`: Provides new evidence chunks.
- `app.session_state`: Provides conversational context and prior claim history; receives updated claims.
- `app.verifier`: Consumes each generated claim to evaluate support before client emission.
- `app.telemetry`: Logs claim synthesis events, latency, and token metrics.

WHY THIS ARCHITECTURE:
- Generating structured claims first turns factual statements into discrete, verifiable data units.
- Prevents hallucinated claims from leaking into the streamed output before verification.
- Enables granular updates and diffing during iterative session refinement.
================================================================================
"""

from typing import List, AsyncGenerator, Dict, Any, Optional
from app.session_state import SessionState, StructuredClaim, RetrievedEvidence
from app.config import Settings, settings
from app.telemetry import TelemetryLogger, telemetry_logger


class StructuredClaimSynthesizer:
    """
    Generates structured factual claims with section citations from evidence and session state.
    """

    def __init__(
        self,
        app_settings: Settings = settings,
        logger: TelemetryLogger = telemetry_logger
    ):
        self.settings = app_settings
        self.telemetry = logger
        # TODO: Initialize LLM client configured for structured claim generation schema

import re
import os
import json
from typing import List, AsyncGenerator, Dict, Any, Optional
from app.session_state import SessionState, StructuredClaim, RetrievedEvidence, ClaimStatus
from app.config import Settings, settings
from app.telemetry import TelemetryLogger, telemetry_logger


class StructuredClaimSynthesizer:
    """
    Generates structured factual claims with section citations from evidence and session state.
    """

    def __init__(
        self,
        app_settings: Settings = settings,
        logger: TelemetryLogger = telemetry_logger
    ):
        self.settings = app_settings
        self.telemetry = logger

    def _format_citation(self, chunk: RetrievedEvidence) -> str:
        """Standardized citation string format."""
        sec = chunk.section_title or "General"
        return f"[DOC: {chunk.doc_id} | Section: {sec} | Chunk: {chunk.chunk_id}]"

    def _extract_fallback_claims(
        self,
        user_query: str,
        evidence: List[RetrievedEvidence],
        session_state: SessionState,
        is_refinement: bool = False
    ) -> List[StructuredClaim]:
        """
        Deterministic, rule-based claim extraction from retrieved evidence when
        running without LLM API credentials or in deterministic test mode.
        """
        claims: List[StructuredClaim] = []
        next_id = 1

        # In refinement mode, retain prior valid claims if available
        if is_refinement and session_state.claims:
            for prior in session_state.claims:
                if prior.status in (ClaimStatus.SUPPORTED, ClaimStatus.PARTIALLY_SUPPORTED):
                    claims.append(StructuredClaim(
                        id=next_id,
                        text=prior.text,
                        cites=list(prior.cites),
                        intent_id=prior.intent_id,
                        status=prior.status
                    ))
                    next_id += 1

        if not evidence:
            # Unsupported / no-evidence query
            claims.append(StructuredClaim(
                id=next_id,
                text=f"The corporate policy knowledge base does not contain verified rules regarding: '{user_query}'.",
                cites=[],
                intent_id=0,
                status=ClaimStatus.UNSUPPORTED,
                verification_reasoning="No relevant evidence chunks found in corporate corpus."
            ))
            return claims

        # Extract sentences from retrieved evidence chunks matching the query
        for chunk in evidence:
            cite_tag = self._format_citation(chunk)
            sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", chunk.text) if s.strip()]
            
            # Select key policy sentences from each chunk
            for sent in sentences[:2]:
                # Clean markdown list markers
                clean_sent = re.sub(r"^[-*•\d.]+\s*", "", sent).strip()
                # Clean leading bold headers if present e.g. "**Flight Duration:** " -> "For flight duration: "
                clean_sent = re.sub(r"^\*\*(.+?):\*\*\s*", r"\1: ", clean_sent)
                
                if len(clean_sent.split()) >= 4:
                    claims.append(StructuredClaim(
                        id=next_id,
                        text=clean_sent,
                        cites=[cite_tag],
                        intent_id=chunk.sub_query_id,
                        status=ClaimStatus.UNVERIFIED
                    ))
                    next_id += 1

        return claims

    async def generate_structured_claims(
        self,
        user_query: str,
        evidence: List[RetrievedEvidence],
        session_state: SessionState,
        is_refinement: bool = False
    ) -> List[StructuredClaim]:
        """
        Calls LLM (or deterministic extractor) to produce a list of structured factual claims.
        """
        api_key = getattr(self.settings, "llm_api_key", "") or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "") or os.getenv("LLM_API_KEY", "")

        if api_key and api_key.strip():
            try:
                from google import genai
                from google.genai import types
                from pydantic import BaseModel, Field

                class ClaimItem(BaseModel):
                    id: int
                    text: str = Field(description="Factual claim sentence directly derived from cited evidence.")
                    cites: List[str] = Field(description="List of citation tags in the format: [DOC: doc_id | Section: section_title | Chunk: chunk_id]")
                    intent: int = Field(default=0, description="Associated sub-query index.")

                class ClaimSynthesisResponse(BaseModel):
                    claims: List[ClaimItem]

                client = genai.Client(api_key=api_key)

                evidence_context = []
                for ev in evidence:
                    tag = self._format_citation(ev)
                    evidence_context.append(f"{tag}\n{ev.text}\n")

                prompt = f"""You are the Structured Claim Synthesis Engine for an Enterprise Live RAG system.
Synthesize atomic factual claims strictly supported by the evidence passages below.
Every claim MUST cite its source using the exact format: [DOC: doc_id | Section: section_title | Chunk: chunk_id].
If the evidence does not contain information to answer the question, do not invent facts; state that information is unavailable.

[Retrieved Evidence]
{"".join(evidence_context)}

[Prior Claims in Session]
{[c.text for c in session_state.claims]}

[User Query]
"{user_query}"
"""
                response = client.models.generate_content(
                    model=self.settings.llm_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ClaimSynthesisResponse,
                        temperature=self.settings.llm_temperature
                    )
                )

                if response.text:
                    parsed = json.loads(response.text)
                    claims_list: List[StructuredClaim] = []
                    for item in parsed.get("claims", []):
                        claims_list.append(StructuredClaim(
                            id=item.get("id", len(claims_list) + 1),
                            text=item.get("text", ""),
                            cites=item.get("cites", []),
                            intent_id=item.get("intent", 0),
                            status=ClaimStatus.UNVERIFIED
                        ))
                    if claims_list:
                        return claims_list
            except Exception:
                pass

        # Fallback deterministic extraction
        return self._extract_fallback_claims(user_query, evidence, session_state, is_refinement)

    async def stream_claims_incrementally(
        self,
        user_query: str,
        evidence: List[RetrievedEvidence],
        session_state: SessionState,
        is_refinement: bool = False
    ) -> AsyncGenerator[StructuredClaim, None]:
        """
        Streams structured claims one by one as they are generated for pipelined verification.
        """
        claims = await self.generate_structured_claims(
            user_query=user_query,
            evidence=evidence,
            session_state=session_state,
            is_refinement=is_refinement
        )
        for claim in claims:
            yield claim

    def render_claims_to_markdown(
        self,
        claims: List[StructuredClaim],
        uncertainty_notes: Optional[List[str]] = None,
        answer_version: int = 1
    ) -> str:
        """
        Renders verified structured claims and citations into a clean markdown response.
        """
        lines: List[str] = []

        # Render verified factual claims
        verified_claims = [c for c in claims if c.status in (ClaimStatus.SUPPORTED, ClaimStatus.PARTIALLY_SUPPORTED)]
        if verified_claims:
            for claim in verified_claims:
                text = claim.softened_text if (claim.status == ClaimStatus.PARTIALLY_SUPPORTED and claim.softened_text) else claim.text
                cite_str = " ".join(claim.cites) if claim.cites else ""
                lines.append(f"- {text} {cite_str}".strip())
        else:
            lines.append("No verified factual policy statements could be grounded from the available evidence.")

        # Render Uncertainty section if unsupported claims or notes exist
        if uncertainty_notes:
            lines.append("\n### ⚠️ Uncertainty & Unverified Inquiries")
            for note in uncertainty_notes:
                lines.append(f"- {note}")

        lines.append(f"\n*(Answer Version: v{answer_version})*")
        return "\n".join(lines)

    async def synthesize_no_retrieval_response(
        self,
        user_query: str,
        session_state: SessionState
    ) -> AsyncGenerator[StructuredClaim, None]:
        """
        Synthesizes response directly from existing SessionState when Controller decides retrieve=False.
        """
        if session_state.claims:
            for c in session_state.claims:
                yield c
        elif session_state.previous_answer:
            yield StructuredClaim(
                id=1,
                text=session_state.previous_answer,
                cites=list(session_state.citations),
                status=ClaimStatus.SUPPORTED
            )
        else:
            yield StructuredClaim(
                id=1,
                text="Hello! How can I assist you with corporate travel, lodging, catering, or event policies today?",
                cites=[],
                status=ClaimStatus.SUPPORTED
            )

