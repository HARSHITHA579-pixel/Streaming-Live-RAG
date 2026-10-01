"""
FastAPI WebSocket and Application Entry Point for Streaming RAG

================================================================================
RESPONSIBILITY:
- Initialize FastAPI web server, WebSocket streaming endpoints, and dependency containers.
- Orchestrate end-to-end streaming lifecycle:
    1. Ingest real-time transcript chunks from WebSocket into an async stream queue (asyncio.Queue).
    2. Invoke Two-Tier Controller (T0 stability gate + T1 multi-intent decision with SessionState context).
    3. Route through No-Retrieval bypass or Multi-Intent Hybrid Retriever.
    4. Pass retrieved evidence to Structured Claim Synthesis.
    5. Stream claims through Claim-Level Verifier one-by-one.
    6. Stream ONLY verified claims, grounded citations, uncertainty notes, and answer version to the client.
- Expose health check and telemetry inspection endpoints.

INPUTS:
- WebSocket messages containing real-time audio transcript chunks or text queries.
- HTTP requests for health check and session telemetry.

OUTPUTS:
- WebSocket streaming responses containing structured JSON events:
  'transcript', 'controller', 'retrieval', 'claim', 'verification', 'answer', 'error'.

CONNECTED COMPONENTS:
- `app.config`: Server and runtime parameters.
- `app.session_state`: Persistent state per active session.
- `app.controller`: 2-Tier gating and intent decomposition.
- `app.retriever`: Multi-intent hybrid search.
- `app.synthesis`: Incremental structured claim generation.
- `app.verifier`: Claim grounding and verification before client emission.
- `app.telemetry`: Side-channel logging of all pipeline events to JSONL.

WHY THIS ARCHITECTURE:
- `main.py` is strictly an orchestration and communication layer.
- Isolating business logic into modular components ensures independent testability and maintainability.
================================================================================
"""

import os
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Awaitable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse

from app.config import settings, Settings
from app.session_state import SessionState, StructuredClaim, RetrievedEvidence, ClaimStatus
from app.controller import TwoTierController, ControllerDecision
from app.retriever import HybridRetriever
from app.synthesis import StructuredClaimSynthesizer
from app.verifier import ClaimVerifier
from app.telemetry import TelemetryLogger, telemetry_logger, TelemetryEvent

logger = logging.getLogger("streaming_rag.main")

# FastAPI Application Instance
app = FastAPI(
    title="Streaming RAG Service",
    description="Live-streaming RAG system with early retrieval, multi-intent decomposition, and claim-level verification.",
    version="0.1.0"
)

# Active session store (session_id -> SessionState)
active_sessions: Dict[str, SessionState] = {}

# Session transcript buffer tracking for T0 delta evaluation (session_id -> previous buffer)
session_transcript_buffers: Dict[str, str] = {}

# Session async queues for incoming streaming chunks (session_id -> asyncio.Queue)
session_queues: Dict[str, asyncio.Queue] = {}


class PipelineOrchestrator:
    """
    Wires together Controller, Retriever, Synthesizer, Verifier, Session State, and Telemetry.
    """

    def __init__(self, app_settings: Settings = settings, logger_instance: TelemetryLogger = telemetry_logger):
        self.settings = app_settings
        self.telemetry = logger_instance
        self.controller = TwoTierController(app_settings, logger_instance)
        self.retriever = HybridRetriever(app_settings, logger_instance)
        self.synthesizer = StructuredClaimSynthesizer(app_settings, logger_instance)
        self.verifier = ClaimVerifier(app_settings, logger_instance)

    async def get_or_create_session(self, session_id: str) -> SessionState:
        """Retrieves or initializes a session's persistent state."""
        if session_id not in active_sessions:
            active_sessions[session_id] = SessionState(session_id=session_id)
        return active_sessions[session_id]

    def evaluate_evidence_relevance_gate(
        self,
        query: str,
        evidence: List[RetrievedEvidence],
        intent_type: str = "REQUIREMENTS",
        sub_queries: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Strong Evidence Relevance & Query-Corpus Compatibility Gate.

        Evaluates retrieved chunks and determines if the evidence is genuinely relevant
        and answerable for the user's intent.

        Distinguishes:
          - Category A (Genuinely Relevant): Directly addresses query subject and required guidelines.
          - Category B (Weakly Related): Mentions generic words ('event', 'safety', 'building', 'food', 'hall', '50')
            without supporting the actual requested operational need.
          - Category C (Irrelevant): Off-topic chunks (rainwater harvesting, high-hazard factory chemicals, etc.).

        Out-of-corpus queries (e.g. venue directories, restaurant listings, hotel searches, phone numbers, booking actions,
        weather forecasts, external vendor pricing) are explicitly gated with status="insufficient_evidence" and should_synthesize=False.
        """
        import re

        clean_q = query.strip().lower()

        # 1. Detect inherently out-of-corpus query classes
        is_policy_inquiry = bool(re.search(r"\b(reimbursement|per diem|rate limit|policy|policies|rules for|guidelines for|regulations)\b", clean_q))

        venue_patterns = [
            r"\b(dining hall|banquet hall|wedding venue|party hall|buffet)\b",
            r"\b(find (?:me )?(?:a )?(?:hotel|restaurant|venue|dining hall|banquet|hall|place))\b",
            r"\b(need (?:a )?(?:dining hall|banquet hall|hotel|restaurant|wedding venue|venue|place to eat))\b",
            r"\b(recommend (?:a )?(?:restaurant|hotel|venue|dining hall|place to eat|buffet))\b",
            r"\b(best restaurant|best hotel|best buffet|cheapest hotel|cheapest venue|top restaurant)\b",
            r"\b(which (?:hotel|restaurant|venue|cafe|buffet))\b",
            r"\b(hotel in [a-z]+|restaurant in [a-z]+|venue in [a-z]+|restaurant near [a-z]+|hotel near [a-z]+|venue near [a-z]+)\b",
            r"\b(where can i (?:rent|find|book|hire) (?:a |me )?(?:hall|venue|hotel|restaurant|room|chairs?|tents?))\b",
            r"\b(hall for \d+|venue for \d+|hotel for \d+|dining hall for \d+|room for rent|rent a hall|rent a venue)\b"
        ]
        is_venue_query = not is_policy_inquiry and any(re.search(p, clean_q) for p in venue_patterns)

        booking_patterns = [
            r"\b(book (?:a |me )?(?:banquet|hall|hotel|table|flight|ticket|venue|seats?|room))\b",
            r"\b(reserve (?:a |me )?(?:table|seats?|hotel|room|hall|venue))\b",
            r"\b(order \d+ (?:box|lunches|meals|pizzas?))\b",
            r"\b(rent chairs|hire event|hire vendor|flight tickets?|book taxi|rental vendor)\b"
        ]
        is_booking_query = not is_policy_inquiry and any(re.search(p, clean_q) for p in booking_patterns)

        unsupported_dir_patterns = [
            r"\b(phone number|contact number|mobile number|email address|price to rent|how much does it cost|weather forecast|weather tomorrow|weather for|3-course dinner|dinner menu|menu for|vendor near me|rental vendor|sound system)\b",
            r"\b(give me the (?:phone|contact) number)\b",
            r"\b(what is the (?:price|cost|weather|phone|menu))\b"
        ]
        is_unsupported_dir = any(re.search(p, clean_q) for p in unsupported_dir_patterns)

        unsupported_topics = ["cryptocurrency", "pet", "bitcoin", "crypto", "drone", "stock market", "shares", "crypto wallet"]
        is_unsupported_topic = any(t in clean_q for t in unsupported_topics)

        # Immediate abstention for out-of-corpus query categories
        if is_venue_query or is_booking_query or is_unsupported_dir or is_unsupported_topic or intent_type in ("LOCATION_VENUE_REQUEST", "BOOKING_ACTION", "UNSUPPORTED"):
            # Construct a clear, domain-specific abstention reason
            if is_venue_query:
                reason = f"The indexed corpus contains event safety and compliance guidelines, but does not contain venue listings or directory information to identify or book a venue for '{query}'."
            elif is_booking_query:
                reason = f"The indexed corpus does not provide booking actions or transaction services for '{query}'."
            elif is_unsupported_dir:
                reason = f"The indexed corpus contains public safety and regulatory guidelines, but does not contain commercial directories, contact details, pricing, or menus for '{query}'."
            else:
                reason = f"The indexed corpus does not contain policy documentation or guidelines regarding: '{query}'."

            return {
                "status": "insufficient_evidence",
                "should_synthesize": False,
                "reason": reason,
                "confidence": 0.95,
                "relevant_evidence": [],
                "weak_evidence": evidence,
                "irrelevant_evidence": [],
                "relevance_scores": {ev.chunk_id: 0.1 for ev in evidence}
            }

        if not evidence:
            return {
                "status": "insufficient_evidence",
                "should_synthesize": False,
                "reason": f"No evidence retrieved from corpus for '{query}'.",
                "confidence": 0.90,
                "relevant_evidence": [],
                "weak_evidence": [],
                "irrelevant_evidence": [],
                "relevance_scores": {}
            }

        # 2. Score each retrieved chunk against specific topical keywords
        stop_words = {
            "what", "is", "the", "are", "and", "for", "of", "in", "to", "a", "an", "on", "by", "with",
            "about", "how", "can", "could", "should", "would", "do", "does", "did", "if", "when",
            "policy", "policies", "rule", "rules", "guideline", "guidelines", "requirement",
            "requirements", "tell", "me", "please", "i", "we", "my", "our", "you", "your", "their",
            "this", "that", "these", "those", "have", "has", "had", "be", "been", "being", "as",
            "at", "from", "or", "any", "all", "some", "which", "there", "where", "who", "whom",
            "need", "large", "give", "mass", "public", "provisions", "measures", "arrangements"
        }
        # Generic corpus tokens that should not inflate relevance score
        generic_tokens = {"event", "events", "safety", "fire", "gathering", "gatherings", "plan", "planning", "guidelines", "structure", "structures", "people", "hall", "building", "standards"}

        query_words = [w for w in re.findall(r"\w+", clean_q) if w not in stop_words]
        specific_query_words = [w for w in query_words if w not in generic_tokens]

        relevant_chunks: List[RetrievedEvidence] = []
        weak_chunks: List[RetrievedEvidence] = []
        irrelevant_chunks: List[RetrievedEvidence] = []
        scores: Dict[str, float] = {}

        for chunk in evidence:
            chunk_text = chunk.text.lower()

            # Target subquery words if decomposed
            sq_text = sub_queries[chunk.sub_query_id] if (sub_queries and 0 <= chunk.sub_query_id < len(sub_queries)) else clean_q
            sq_words = [w for w in re.findall(r"\w+", sq_text.lower()) if w not in stop_words]
            sq_spec_words = [w for w in sq_words if w not in generic_tokens]

            spec_matches = sum(1 for w in specific_query_words if w in chunk_text)
            sq_spec_matches = sum(1 for w in sq_spec_words if w in chunk_text)
            sq_all_matches = sum(1 for w in sq_words if w in chunk_text)
            all_matches = sum(1 for w in query_words if w in chunk_text)

            # Check off-topic indicators in chunk (rainwater harvesting, solar heating, high hazard chemical)
            is_off_topic = False
            if "rainwater" in chunk_text or "solar water" in chunk_text or "hazardous chemical" in chunk_text:
                if not any(w in clean_q for w in ["rainwater", "solar", "chemical"]):
                    is_off_topic = True

            if is_off_topic:
                score = 0.05
                irrelevant_chunks.append(chunk)
            elif sq_spec_matches >= 1 or spec_matches >= 1:
                # Genuinely relevant (Category A)
                score = min(1.0, 0.6 + 0.2 * max(spec_matches, sq_spec_matches))
                relevant_chunks.append(chunk)
            elif (not specific_query_words or not sq_spec_words) and (sq_all_matches >= 1 or all_matches >= 1):
                # Query only had general words (e.g. fire safety requirements)
                score = 0.7
                relevant_chunks.append(chunk)
            elif all_matches >= 1:
                # Category B (Weakly related)
                score = 0.3
                weak_chunks.append(chunk)
            else:
                # Category C (Irrelevant)
                score = 0.0
                irrelevant_chunks.append(chunk)

            scores[chunk.chunk_id] = score

        if not relevant_chunks:
            return {
                "status": "insufficient_evidence",
                "should_synthesize": False,
                "reason": f"The indexed corpus does not contain sufficient evidence to answer: '{query}'.",
                "confidence": 0.85,
                "relevant_evidence": [],
                "weak_evidence": weak_chunks,
                "irrelevant_evidence": irrelevant_chunks,
                "relevance_scores": scores
            }

        # Select balanced Category A chunks across sub-queries (max 4 total)
        sq_groups: Dict[int, List[RetrievedEvidence]] = {}
        for c in relevant_chunks:
            sq_groups.setdefault(c.sub_query_id, []).append(c)

        top_relevant: List[RetrievedEvidence] = []
        if sq_groups:
            # Round-robin pick from each sub-query group to guarantee multi-intent coverage
            max_per_sq = max(1, 4 // len(sq_groups))
            for sq_id in sorted(sq_groups.keys()):
                top_relevant.extend(sq_groups[sq_id][:max_per_sq])
            # If still room, fill with remaining top relevant chunks
            for c in relevant_chunks:
                if len(top_relevant) >= 4:
                    break
                if c not in top_relevant:
                    top_relevant.append(c)
        else:
            top_relevant = relevant_chunks[:4]

        return {
            "status": "supported",
            "should_synthesize": True,
            "reason": "Sufficient genuine evidence identified in indexed corpus.",
            "confidence": 0.92,
            "relevant_evidence": top_relevant,
            "weak_evidence": weak_chunks,
            "irrelevant_evidence": irrelevant_chunks,
            "relevance_scores": scores
        }

    async def process_streaming_transcript(
        self,
        session_id: str,
        transcript_chunk: str,
        websocket: Optional[WebSocket] = None,
        event_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        is_final: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Processes an incoming transcript chunk through the streaming RAG pipeline.

        CRITICAL STREAMING ORDER:
        1. Ingest transcript -> Emit 'transcript' event.
        2. T0 Gate & T1 Decision -> Emit 'controller' event.
           - If T0 says unstable -> WAIT (no retrieval, no synthesis, no answer emission).
        3. If should_retrieve is False (conversational / formatting bypass):
           - If is_final is False -> return early without emitting answer.
           - If is_final is True -> Synthesize directly from session state -> Verify -> Emit 'claim', 'verification', 'answer'.
        4. If should_retrieve is True:
           - Multi-intent hybrid retrieval (BM25 + Dense + per-sub-query RRF + Union + Dedup + Min Evidence).
           - Update session state evidence pool.
           - Emit 'retrieval' event.
           - If is_final is False (background retrieval while user is still speaking):
             - Stop here and return early (evidence is safely stored in evidence pool).
           - If is_final is True (user explicitly stopped or final turn):
             - Synthesize structured claims from accumulated evidence pool -> Emit 'claim' events.
             - Claim-by-claim verification -> Emit 'verification' events.
             - Stream final verified grounded answer, citations, uncertainty notes, and answer version.
        """
        t_total_start = time.perf_counter()
        events_emitted: List[Dict[str, Any]] = []

        async def emit(event_data: Dict[str, Any]) -> None:
            events_emitted.append(event_data)
            if websocket is not None:
                try:
                    await websocket.send_json(event_data)
                except Exception:
                    pass
            if event_callback is not None:
                try:
                    await event_callback(event_data)
                except Exception:
                    pass

        clean_text = transcript_chunk.strip()
        if not clean_text:
            return events_emitted

        session_state = await self.get_or_create_session(session_id)
        previous_buffer = session_transcript_buffers.get(session_id, "")

        # 1. Emit Transcript Event
        await emit({
            "event": "transcript",
            "session_id": session_id,
            "text": clean_text,
            "is_final": is_final
        })

        # 2. Controller Evaluation (T0 Stability Gate + T1 Multi-Intent Routing)
        t_ctrl_start = time.perf_counter()
        decision: ControllerDecision = await self.controller.process_incoming_transcript(
            current_transcript=clean_text,
            previous_transcript=previous_buffer,
            session_state=session_state
        )
        ctrl_latency_ms = (time.perf_counter() - t_ctrl_start) * 1000

        # Debug Logging
        t0_action_str = "PROCEED" if decision.t0_stable else "WAIT"
        logger.info(
            f"[DEBUG-TRACE] session_id={session_id} | turn={session_state.answer_version} | "
            f"transcript='{clean_text}' | t0_stable={decision.t0_stable} | t0_action={t0_action_str} | "
            f"t1_should_retrieve={decision.should_retrieve} | timestamp={time.time()}"
        )
        print(
            f"[DEBUG-TRACE] session_id={session_id} | turn={session_state.answer_version} | "
            f"transcript='{clean_text}' | t0_stable={decision.t0_stable} | t0_action={t0_action_str} | "
            f"t1_should_retrieve={decision.should_retrieve} | timestamp={time.time()}"
        )

        await emit({
            "event": "controller",
            "session_id": session_id,
            "t0_stable": decision.t0_stable,
            "should_retrieve": decision.should_retrieve,
            "is_refine": decision.is_refine,
            "refinement_type": decision.refinement_type,
            "intent_type": decision.intent_type,
            "is_supersession": decision.is_supersession,
            "is_removal": decision.is_removal,
            "is_continuation": decision.is_continuation,
            "sub_queries": decision.sub_queries,
            "reasoning": decision.reasoning,
            "latency_ms": ctrl_latency_ms,
            "is_final": is_final
        })

        # If T0 Gate determines the utterance is incomplete or drifting -> WAIT
        if not decision.t0_stable:
            session_transcript_buffers[session_id] = clean_text
            self.telemetry.log_t0_gate(
                session_id=session_id,
                delta_similarity=0.0,
                is_stable=False,
                latency_ms=ctrl_latency_ms,
                decision_action="WAIT",
                answer_version=session_state.answer_version
            )
            return events_emitted

        # Update stable transcript buffer for this session
        session_transcript_buffers[session_id] = clean_text
        self.telemetry.log_t0_gate(
            session_id=session_id,
            delta_similarity=1.0,
            is_stable=True,
            latency_ms=ctrl_latency_ms,
            decision_action="PROCEED",
            answer_version=session_state.answer_version
        )
        self.telemetry.log_t1_decision(
            session_id=session_id,
            retrieve=decision.should_retrieve,
            is_refine=decision.is_refine,
            sub_queries=decision.sub_queries,
            answer_version=session_state.answer_version,
            latency_ms=ctrl_latency_ms,
            reasoning=decision.reasoning,
            decision_action="REFINE" if decision.is_refine else ("RETRIEVE" if decision.should_retrieve else "NO_RETRIEVAL")
        )

        # 3. Path A: No-Retrieval Bypass (Conversational / Acknowledgement / Formatting)
        if not decision.should_retrieve:
            if not is_final:
                # Still recording, conversational segment does not need early retrieval
                return events_emitted

            t_synth_start = time.perf_counter()
            claims_for_turn: List[StructuredClaim] = []
            total_ver_latency_ms = 0.0

            is_conversational = bool(decision.reasoning and "conversational" in decision.reasoning.lower())
            if is_conversational:
                conv_claim = StructuredClaim(
                    id=1,
                    text="Hi! How can I help you?",
                    cites=[],
                    status=ClaimStatus.SUPPORTED,
                    verification_reasoning="Conversational response"
                )
                await emit({
                    "event": "claim",
                    "session_id": session_id,
                    "claim_id": conv_claim.id,
                    "text": conv_claim.text,
                    "cites": [],
                    "intent_id": 0
                })
                await emit({
                    "event": "verification",
                    "session_id": session_id,
                    "claim_id": conv_claim.id,
                    "status": conv_claim.status.value,
                    "text": conv_claim.text,
                    "reasoning": "Conversational bypass",
                    "cites": [],
                    "latency_ms": 0.0
                })
                claims_for_turn.append(conv_claim)
            else:
                async for raw_claim in self.synthesizer.synthesize_no_retrieval_response(clean_text, session_state):
                    await emit({
                        "event": "claim",
                        "session_id": session_id,
                        "claim_id": raw_claim.id,
                        "text": raw_claim.text,
                        "cites": raw_claim.cites,
                        "intent_id": raw_claim.intent_id
                    })

                    t_ver_start = time.perf_counter()
                    verified_claim = await self.verifier.verify_single_claim(raw_claim, session_state)
                    ver_latency_ms = (time.perf_counter() - t_ver_start) * 1000
                    total_ver_latency_ms += ver_latency_ms

                    await emit({
                        "event": "verification",
                        "session_id": session_id,
                        "claim_id": verified_claim.id,
                        "status": verified_claim.status.value,
                        "text": verified_claim.softened_text or verified_claim.text,
                        "reasoning": verified_claim.verification_reasoning,
                        "cites": verified_claim.cites,
                        "latency_ms": ver_latency_ms
                    })
                    claims_for_turn.append(verified_claim)

            synth_latency_ms = (time.perf_counter() - t_synth_start) * 1000
            session_state.record_refinement(clean_text)
            session_state.update_claims(claims_for_turn)
            answer_text = self.synthesizer.render_claims_to_markdown(
                claims=claims_for_turn,
                uncertainty_notes=session_state.uncertainty_notes,
                answer_version=session_state.answer_version,
                session_state=session_state
            )
            session_state.previous_answer = answer_text

            self.telemetry.log_synthesis(
                session_id=session_id,
                latency_ms=synth_latency_ms,
                claims_generated=len(claims_for_turn),
                citations_generated=0,
                answer_version=session_state.answer_version,
                is_refinement=False,
                usage_available=False
            )

            total_latency_ms = (time.perf_counter() - t_total_start) * 1000
            self.telemetry.log_e2e(
                session_id=session_id,
                total_latency_ms=total_latency_ms,
                controller_latency_ms=ctrl_latency_ms,
                retrieval_latency_ms=0.0,
                synthesis_latency_ms=synth_latency_ms,
                verification_latency_ms=total_ver_latency_ms,
                answer_version=session_state.answer_version
            )

            await emit({
                "event": "answer",
                "session_id": session_id,
                "answer_version": session_state.answer_version,
                "answer": answer_text,
                "citations": sorted(list(session_state.citations)),
                "uncertainty": session_state.uncertainty_notes,
                "is_refinement": False,
                "total_latency_ms": total_latency_ms
            })
            return events_emitted

        # 4. Path B: Multi-Intent Retrieval & Grounded Synthesis Path
        t_ret_start = time.perf_counter()
        sub_queries = decision.sub_queries if decision.sub_queries else [clean_text]
        evidence: List[RetrievedEvidence] = await self.retriever.retrieve_for_sub_queries(
            sub_queries=sub_queries,
            session_id=session_id
        )
        ret_latency_ms = (time.perf_counter() - t_ret_start) * 1000

        # Evaluate Evidence Relevance Gate & Query-Corpus Compatibility
        gate_result = self.evaluate_evidence_relevance_gate(clean_text, evidence, decision.intent_type, sub_queries=sub_queries)
        filtered_evidence = gate_result["relevant_evidence"]

        # Update Session State Evidence Pool & Intents with Supersession Awareness
        if decision.is_supersession:
            active_target = ", ".join(sub_queries) if sub_queries else clean_text
            session_state.supersede_active_context(new_intent=active_target)
            if decision.superseded_terms:
                for st in decision.superseded_terms:
                    if st not in session_state.superseded_intents:
                        session_state.superseded_intents.append(st)
        elif decision.is_removal and decision.superseded_terms:
            session_state.remove_intent_context(decision.superseded_terms[0])
        elif not decision.is_refine and not decision.is_continuation:
            # Brand new top-level intent/topic: isolate active pool so previous unrelated topic evidence doesn't pollute synthesis
            active_target = ", ".join(sub_queries) if sub_queries else clean_text
            session_state.supersede_active_context(new_intent=active_target)

        if filtered_evidence:
            session_state.add_evidence(filtered_evidence, is_supersession=False)

        for sq in sub_queries:
            if sq not in session_state.active_intents:
                session_state.active_intents.append(sq)
            if sq not in session_state.covered_intents:
                session_state.covered_intents.append(sq)

        self.telemetry.log_retrieval(
            session_id=session_id,
            latency_ms=ret_latency_ms,
            number_of_subqueries=len(sub_queries),
            evidence_count=len(filtered_evidence),
            retrieval_call_count=len(sub_queries),
            answer_version=session_state.answer_version
        )

        await emit({
            "event": "retrieval",
            "session_id": session_id,
            "sub_queries": sub_queries,
            "is_refinement": decision.is_refine,
            "refinement_type": decision.refinement_type,
            "evidence_count": len(filtered_evidence),
            "gate_status": gate_result["status"],
            "should_synthesize": gate_result["should_synthesize"],
            "chunks": [
                {
                    "chunk_id": ev.chunk_id,
                    "doc_id": ev.doc_id,
                    "section": ev.section_title,
                    "score": gate_result["relevance_scores"].get(ev.chunk_id, ev.score),
                    "sub_query_id": ev.sub_query_id,
                    "retrieval_method": ev.retrieval_method
                }
                for ev in filtered_evidence
            ],
            "latency_ms": ret_latency_ms,
            "is_final": is_final
        })

        # If user is still speaking in this voice turn, stop after background retrieval
        if not is_final:
            return events_emitted

        # Final turn: record refinement to advance answer version
        session_state.record_refinement(clean_text, is_supersession=decision.is_supersession)

        # Handle Abstention / Insufficient Evidence Gate
        if not gate_result["should_synthesize"] or not filtered_evidence:
            t_synth_start = time.perf_counter()
            abstention_claims: List[StructuredClaim] = []

            # Check if this is a commercial venue / booking / directory query vs generic unsupported query
            is_venue_or_dir = decision.intent_type in ("LOCATION_VENUE_REQUEST", "BOOKING_ACTION", "UNSUPPORTED") or any(
                term in clean_text.lower() for term in ["dining hall", "hotel", "restaurant", "banquet", "venue", "book", "reserve", "phone", "price", "weather", "menu"]
            )

            if is_venue_or_dir:
                abstention_claims.append(StructuredClaim(
                    id=1,
                    text=gate_result["reason"],
                    cites=[],
                    intent_id=0,
                    status=ClaimStatus.SUPPORTED,
                    verification_reasoning="Out-of-corpus abstention gate"
                ))
                abstention_claims.append(StructuredClaim(
                    id=2,
                    text="I can provide verified guidelines on event safety standards, food hygiene, accessibility, crowd management, and emergency preparedness covered in the indexed documents.",
                    cites=[],
                    intent_id=0,
                    status=ClaimStatus.SUPPORTED,
                    verification_reasoning="Corpus boundary guidance"
                ))
            else:
                abstention_claims.append(StructuredClaim(
                    id=1,
                    text=f"The event planning and safety compliance knowledge base does not contain verified rules regarding: '{clean_text}'.",
                    cites=[],
                    intent_id=0,
                    status=ClaimStatus.UNSUPPORTED,
                    verification_reasoning="Insufficient evidence in corpus"
                ))

            synth_latency_ms = (time.perf_counter() - t_synth_start) * 1000

            self.telemetry.log_synthesis(
                session_id=session_id,
                latency_ms=synth_latency_ms,
                claims_generated=len(abstention_claims),
                citations_generated=0,
                answer_version=session_state.answer_version,
                is_refinement=decision.is_refine,
                usage_available=False
            )

            for claim in abstention_claims:
                self.telemetry.log_claim_verification(
                    session_id=session_id,
                    claim_id=claim.id,
                    claim_text=claim.text,
                    result=claim.status.value,
                    citations=[],
                    method="deterministic_abstention_gate",
                    latency_ms=0.0,
                    answer_version=session_state.answer_version
                )
                await emit({
                    "event": "claim",
                    "session_id": session_id,
                    "claim_id": claim.id,
                    "text": claim.text,
                    "cites": claim.cites,
                    "intent_id": claim.intent_id
                })
                await emit({
                    "event": "verification",
                    "session_id": session_id,
                    "claim_id": claim.id,
                    "status": claim.status.value,
                    "text": claim.text,
                    "reasoning": claim.verification_reasoning,
                    "cites": [],
                    "latency_ms": 0.0
                })

            uncertainty_note = f"The requested topic '{clean_text}' is not covered in the indexed corporate and event policy corpus."
            session_state.uncertainty_notes = [uncertainty_note]

            session_state.set_active_claims(abstention_claims)
            final_answer = self.synthesizer.render_claims_to_markdown(
                claims=abstention_claims,
                uncertainty_notes=session_state.uncertainty_notes if not is_venue_or_dir else None,
                answer_version=session_state.answer_version
            )
            session_state.previous_answer = final_answer

            total_latency_ms = (time.perf_counter() - t_total_start) * 1000
            self.telemetry.log_e2e(
                session_id=session_id,
                total_latency_ms=total_latency_ms,
                controller_latency_ms=ctrl_latency_ms,
                retrieval_latency_ms=ret_latency_ms,
                synthesis_latency_ms=synth_latency_ms,
                verification_latency_ms=0.0,
                answer_version=session_state.answer_version
            )

            await emit({
                "event": "answer",
                "session_id": session_id,
                "answer_version": session_state.answer_version,
                "answer": final_answer,
                "citations": [],
                "uncertainty": session_state.uncertainty_notes,
                "is_refinement": decision.is_refine,
                "gate_status": gate_result["status"],
                "total_latency_ms": total_latency_ms
            })
            return events_emitted

        # Select only active evidence for synthesis (excludes superseded / stale context)
        synthesis_evidence = list(session_state.active_evidence_pool.values()) if session_state.active_evidence_pool else filtered_evidence

        # 5. Synthesize Structured Claims
        t_synth_start = time.perf_counter()
        generated_claims = await self.synthesizer.generate_structured_claims(
            user_query=clean_text,
            evidence=synthesis_evidence,
            session_state=session_state,
            is_refinement=decision.is_refine
        )
        synth_latency_ms = (time.perf_counter() - t_synth_start) * 1000

        self.telemetry.log_synthesis(
            session_id=session_id,
            latency_ms=synth_latency_ms,
            claims_generated=len(generated_claims),
            citations_generated=sum(len(c.cites) for c in generated_claims),
            answer_version=session_state.answer_version,
            is_refinement=decision.is_refine,
            usage_available=False
        )

        # 6. Stream & Verify Claims Individually
        verified_claims_for_turn: List[StructuredClaim] = []
        turn_uncertainty: List[str] = []
        total_ver_latency_ms = 0.0

        for raw_claim in generated_claims:
            await emit({
                "event": "claim",
                "session_id": session_id,
                "claim_id": raw_claim.id,
                "text": raw_claim.text,
                "cites": raw_claim.cites,
                "intent_id": raw_claim.intent_id
            })

            t_ver_start = time.perf_counter()
            verified_claim = await self.verifier.verify_single_claim(raw_claim, session_state)
            ver_latency_ms = (time.perf_counter() - t_ver_start) * 1000
            total_ver_latency_ms += ver_latency_ms

            await emit({
                "event": "verification",
                "session_id": session_id,
                "claim_id": verified_claim.id,
                "status": verified_claim.status.value,
                "text": verified_claim.softened_text or verified_claim.text,
                "reasoning": verified_claim.verification_reasoning,
                "cites": verified_claim.cites,
                "latency_ms": ver_latency_ms
            })

            if verified_claim.status in (ClaimStatus.SUPPORTED, ClaimStatus.PARTIALLY_SUPPORTED):
                verified_claims_for_turn.append(verified_claim)
            else:
                reason = verified_claim.verification_reasoning or "Lacks grounding in cited evidence."
                turn_uncertainty.append(f"Excluded ungrounded assertion: '{verified_claim.text}' ({reason})")

        # Fallback uncertainty note if query returned no verified facts
        if not synthesis_evidence or not verified_claims_for_turn:
            if not any(u for u in turn_uncertainty):
                turn_uncertainty.append(f"No verified corporate policy found regarding: '{clean_text}'.")

        # Uncertainty notes belong strictly to the current turn
        session_state.uncertainty_notes = list(turn_uncertainty)

        session_state.set_active_claims(verified_claims_for_turn)

        # 7. Render Final Grounded Answer
        claims_to_render = verified_claims_for_turn
        final_answer = self.synthesizer.render_claims_to_markdown(
            claims=claims_to_render,
            uncertainty_notes=session_state.uncertainty_notes,
            answer_version=session_state.answer_version,
            session_state=session_state
        )
        session_state.previous_answer = final_answer

        total_latency_ms = (time.perf_counter() - t_total_start) * 1000
        self.telemetry.log_e2e(
            session_id=session_id,
            total_latency_ms=total_latency_ms,
            controller_latency_ms=ctrl_latency_ms,
            retrieval_latency_ms=ret_latency_ms,
            synthesis_latency_ms=synth_latency_ms,
            verification_latency_ms=total_ver_latency_ms,
            answer_version=session_state.answer_version
        )

        await emit({
            "event": "answer",
            "session_id": session_id,
            "answer_version": session_state.answer_version,
            "answer": final_answer,
            "citations": sorted(list(session_state.citations)),
            "uncertainty": session_state.uncertainty_notes,
            "is_refinement": decision.is_refine,
            "total_latency_ms": total_latency_ms
        })

        return events_emitted


# Global orchestrator instance
orchestrator = PipelineOrchestrator()


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "streaming-rag", "version": "0.1.0"}


@app.get("/", response_class=HTMLResponse)
@app.get("/nexa", response_class=HTMLResponse)
async def get_nexa_app() -> HTMLResponse:
    """Serves the NEXA user-facing Streaming Live RAG assistant."""
    index_path = Path(__file__).resolve().parent.parent / "dashboard" / "index.html"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>NEXA UI not found</h1>", status_code=404)


@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard() -> HTMLResponse:
    """Serves the real-time Streaming RAG observability timeline dashboard."""
    dashboard_path = Path(__file__).resolve().parent.parent / "dashboard" / "timeline.html"
    if dashboard_path.exists():
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Dashboard not found</h1>", status_code=404)


@app.get("/session/{session_id}")
async def get_session_info(session_id: str) -> JSONResponse:
    """Inspects the current state of an active session."""
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    return JSONResponse(content={
        "session_id": session.session_id,
        "answer_version": session.answer_version,
        "covered_intents": session.covered_intents,
        "evidence_count": len(session.evidence_pool),
        "claims_count": len(session.claims),
        "citations": list(session.citations),
        "previous_answer": session.previous_answer,
        "uncertainty_notes": session.uncertainty_notes
    })


@app.post("/session/{session_id}/reset")
async def reset_session(session_id: str) -> JSONResponse:
    """Explicitly resets an active session's state."""
    if session_id in active_sessions:
        active_sessions[session_id].reset()
    if session_id in session_transcript_buffers:
        session_transcript_buffers[session_id] = ""
    return JSONResponse(content={"session_id": session_id, "status": "reset_successful"})


@app.get("/telemetry/{session_id}")
async def get_session_telemetry(session_id: str) -> JSONResponse:
    """
    Retrieves the raw JSONL telemetry records for a specific session for debugging and dashboarding.
    """
    log_path = telemetry_logger.get_session_log_path(session_id)
    if not log_path.exists():
        raise HTTPException(status_code=404, detail=f"No telemetry log found for session {session_id}")

    events = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    events.append(json.loads(line.strip()))
                except Exception:
                    pass
    return JSONResponse(content={"session_id": session_id, "events": events, "count": len(events)})


async def _session_queue_worker(session_id: str, queue: asyncio.Queue, websocket: WebSocket) -> None:
    """
    Background worker that consumes transcript chunks sequentially from the async session queue.
    """
    while True:
        try:
            raw_data = await queue.get()
            if raw_data is None:
                # Sentinel to stop worker
                break

            # Handle JSON payload or raw text string
            transcript_text = ""
            is_final = True
            if isinstance(raw_data, str):
                try:
                    parsed = json.loads(raw_data)
                    if isinstance(parsed, dict):
                        if parsed.get("action") == "reset":
                            if session_id in active_sessions:
                                active_sessions[session_id].reset()
                            session_transcript_buffers[session_id] = ""
                            await websocket.send_json({"event": "reset", "session_id": session_id})
                            continue
                        transcript_text = parsed.get("text") or parsed.get("transcript") or parsed.get("chunk") or ""
                        if "is_final" in parsed:
                            is_final = bool(parsed.get("is_final"))
                    else:
                        transcript_text = raw_data
                except json.JSONDecodeError:
                    transcript_text = raw_data
            elif isinstance(raw_data, dict):
                transcript_text = raw_data.get("text") or raw_data.get("transcript") or raw_data.get("chunk") or ""
                if "is_final" in raw_data:
                    is_final = bool(raw_data.get("is_final"))

            if transcript_text and transcript_text.strip():
                await orchestrator.process_streaming_transcript(
                    session_id=session_id,
                    transcript_chunk=transcript_text,
                    websocket=websocket,
                    is_final=is_final
                )

        except WebSocketDisconnect:
            break
        except Exception as e:
            try:
                await websocket.send_json({
                    "event": "error",
                    "session_id": session_id,
                    "message": str(e)
                })
            except Exception:
                pass
        finally:
            queue.task_done()


@app.websocket("/ws/stream/{session_id}")
async def websocket_transcript_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time live streaming transcript ingestion and verified response streaming.

    FLOW:
    1. Accept WebSocket connection.
    2. Create an isolated `asyncio.Queue` for this session.
    3. Start background queue worker `_session_queue_worker`.
    4. Ingest incoming transcript chunks and put them onto the queue.
    5. Clean up gracefully on disconnect.
    """
    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue()
    session_queues[session_id] = queue

    worker_task = asyncio.create_task(
        _session_queue_worker(session_id=session_id, queue=queue, websocket=websocket)
    )

    try:
        while True:
            data = await websocket.receive_text()
            if data:
                await queue.put(data)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning(f"WebSocket session {session_id} error: {e}")
    finally:
        # Cleanup
        await queue.put(None)  # Signal worker to stop
        worker_task.cancel()
        session_queues.pop(session_id, None)
        try:
            await websocket.close()
        except Exception:
            pass
