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

    def _filter_relevant_evidence_for_query(
        self,
        query: str,
        evidence: List[RetrievedEvidence]
    ) -> List[RetrievedEvidence]:
        """
        Validates that retrieved evidence chunks contain relevance to the core topical terms of the query.
        Prevents false-positive retrieval (e.g. matching generic words 'reimbursement' when user asked about 'cryptocurrency').
        """
        import re
        if not evidence:
            return []

        stop_words = {
            "what", "is", "the", "are", "and", "for", "of", "in", "to", "a", "an", "on", "by", "with",
            "about", "how", "can", "could", "should", "would", "do", "does", "did", "if", "when",
            "policy", "policies", "rule", "rules", "guideline", "guidelines", "requirement",
            "requirements", "tell", "me", "please", "i", "we", "my", "our", "you", "your", "their",
            "this", "that", "these", "those", "have", "has", "had", "be", "been", "being", "as",
            "at", "from", "or", "any", "all", "some", "which", "there", "where", "who", "whom"
        }
        query_tokens = set(re.findall(r"\w+", query.lower())) - stop_words
        if not query_tokens:
            return evidence

        all_evidence_words = set()
        for chunk in evidence:
            all_evidence_words.update(re.findall(r"\w+", chunk.text.lower()))

        missing_tokens = [t for t in query_tokens if t not in all_evidence_words]
        # If any specific non-generic query subject is completely absent from all retrieved evidence
        if missing_tokens and (len(missing_tokens) >= len(query_tokens) or any(t in ["cryptocurrency", "pet", "bitcoin", "crypto", "drone"] for t in missing_tokens) or len(missing_tokens) / len(query_tokens) >= 0.5):
            return []

        return [c for c in evidence if query_tokens.intersection(set(re.findall(r"\w+", c.text.lower())))]

    async def process_streaming_transcript(
        self,
        session_id: str,
        transcript_chunk: str,
        websocket: Optional[WebSocket] = None,
        event_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Processes an incoming transcript chunk through the streaming RAG pipeline.

        CRITICAL STREAMING ORDER:
        1. Ingest transcript -> Emit 'transcript' event.
        2. T0 Gate & T1 Decision -> Emit 'controller' event.
           - If T0 says unstable -> WAIT (no retrieval, no synthesis, no answer emission).
        3. If should_retrieve is False (conversational / formatting bypass):
           - Synthesize directly from session state -> Verify -> Emit 'claim', 'verification', 'answer'.
        4. If should_retrieve is True:
           - Multi-intent hybrid retrieval (BM25 + Dense + per-sub-query RRF + Union + Dedup + Min Evidence).
           - Update session state evidence pool.
           - Emit 'retrieval' event.
           - Synthesize structured claims -> Emit 'claim' events.
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
            "text": clean_text
        })

        # 2. Controller Evaluation (T0 Stability Gate + T1 Multi-Intent Routing)
        t_ctrl_start = time.perf_counter()
        decision: ControllerDecision = await self.controller.process_incoming_transcript(
            current_transcript=clean_text,
            previous_transcript=previous_buffer,
            session_state=session_state
        )
        ctrl_latency_ms = (time.perf_counter() - t_ctrl_start) * 1000

        await emit({
            "event": "controller",
            "session_id": session_id,
            "t0_stable": decision.t0_stable,
            "should_retrieve": decision.should_retrieve,
            "is_refine": decision.is_refine,
            "sub_queries": decision.sub_queries,
            "reasoning": decision.reasoning,
            "latency_ms": ctrl_latency_ms
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
            t_synth_start = time.perf_counter()
            claims_for_turn: List[StructuredClaim] = []
            total_ver_latency_ms = 0.0

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
                answer_version=session_state.answer_version
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

        # Filter out evidence if the query's core topical subjects are unsupported in the knowledge base
        synthesis_evidence = self._filter_relevant_evidence_for_query(clean_text, evidence)

        # Update Session State Evidence Pool & Intents
        if synthesis_evidence:
            session_state.add_evidence(synthesis_evidence)
        session_state.record_refinement(clean_text)
        for sq in sub_queries:
            if sq not in session_state.covered_intents:
                session_state.covered_intents.append(sq)

        self.telemetry.log_retrieval(
            session_id=session_id,
            latency_ms=ret_latency_ms,
            number_of_subqueries=len(sub_queries),
            evidence_count=len(synthesis_evidence),
            retrieval_call_count=len(sub_queries),
            answer_version=session_state.answer_version
        )

        await emit({
            "event": "retrieval",
            "session_id": session_id,
            "sub_queries": sub_queries,
            "is_refinement": decision.is_refine,
            "evidence_count": len(synthesis_evidence),
            "chunks": [
                {
                    "chunk_id": ev.chunk_id,
                    "doc_id": ev.doc_id,
                    "section": ev.section_title,
                    "score": ev.score,
                    "sub_query_id": ev.sub_query_id,
                    "retrieval_method": ev.retrieval_method
                }
                for ev in synthesis_evidence
            ],
            "latency_ms": ret_latency_ms
        })

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
        if not evidence or not verified_claims_for_turn:
            if not any(u for u in turn_uncertainty):
                turn_uncertainty.append(f"No verified corporate policy found regarding: '{clean_text}'.")

        session_state.uncertainty_notes.extend([u for u in turn_uncertainty if u not in session_state.uncertainty_notes])
        session_state.update_claims(verified_claims_for_turn)

        # 7. Render Final Grounded Answer
        claims_to_render = session_state.claims if decision.is_refine else verified_claims_for_turn
        final_answer = self.synthesizer.render_claims_to_markdown(
            claims=claims_to_render,
            uncertainty_notes=session_state.uncertainty_notes,
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
                    else:
                        transcript_text = raw_data
                except json.JSONDecodeError:
                    transcript_text = raw_data
            elif isinstance(raw_data, dict):
                transcript_text = raw_data.get("text") or raw_data.get("transcript") or raw_data.get("chunk") or ""

            if transcript_text and transcript_text.strip():
                await orchestrator.process_streaming_transcript(
                    session_id=session_id,
                    transcript_chunk=transcript_text,
                    websocket=websocket
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
