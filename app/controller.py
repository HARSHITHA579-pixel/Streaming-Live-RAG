"""
Two-Tier Controller for Streaming RAG

================================================================================
RESPONSIBILITY:
- T0 (Intent Stability Gate): Evaluate incoming transcript deltas using cheap embedding similarity
  to determine if user intent is stable enough before triggering expensive LLM reasoning.
- T1 (Multi-Intent Decomposition & Router): Perform a single structured LLM call that decides:
    - retrieve (bool): Whether corpus retrieval is necessary.
    - is_refine (bool): Whether the current turn is a refinement of existing answer/state.
    - sub_queries (List[str]): Decomposed atomic search queries for distinct intents.
- Session State Awareness: T1 MUST inspect `SessionState` (covered_intents, previous claims,
  previous answer, existing evidence) before deciding, ensuring refinement queries target ONLY
  new or modified information.
- No-Retrieval Path: Provide bypass when retrieve=False (e.g., formatting requests or conversational turns),
  routing directly to synthesis using existing session state without corpus queries.

INPUTS:
- Incoming transcript chunk / buffer string from the async stream queue.
- Current `SessionState` object.

OUTPUTS:
- `ControllerDecision` containing `retrieve`, `is_refine`, `sub_queries`, and rationale.

CONNECTED COMPONENTS:
- `app.session_state`: Reads covered intents, previous claims, and answer versions; registers new intents.
- `app.retriever`: Receives decomposed `sub_queries` if `retrieve=True`.
- `app.synthesis`: Receives execution control directly if `retrieve=False`.
- `app.telemetry`: Emits T0 stability checks and T1 decisions to session logs.

WHY THIS ARCHITECTURE:
- T0 prevents spamming LLM calls on every partial audio transcription syllable/word.
- Reading session state in T1 eliminates duplicate retrievals during iterative conversations
  (e.g., searching only for late-booking rules if international travel is already established).
================================================================================
"""

import os
import re
import json
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, cast

from app.session_state import SessionState
from app.config import Settings, settings
from app.telemetry import TelemetryLogger, telemetry_logger


@dataclass
class ControllerDecision:
    """
    Structured decision output produced by Tier 1 (T1) of the controller.
    """
    should_retrieve: bool
    is_refine: bool
    sub_queries: List[str] = field(default_factory=list)
    reasoning: Optional[str] = None
    t0_stable: bool = True

    @property
    def retrieve(self) -> bool:
        """Alias for should_retrieve for compatibility across calling conventions."""
        return self.should_retrieve


class TwoTierController:
    """
    Implements the two-tier gating and multi-intent decomposition controller:
    - T0: Lightweight embedding-delta & lexical stability gate (decides whether to WAIT or proceed).
    - T1: Session-aware structured routing and multi-intent decomposition.
    """

    def __init__(
        self,
        app_settings: Settings = settings,
        logger: TelemetryLogger = telemetry_logger
    ):
        self.settings = app_settings
        self.telemetry = logger
        self.vectorizer: Optional[Any] = None

    def _get_vectorizer(self) -> Any:
        """Lazy initialization of the local hashing vectorizer matching build_index.py."""
        if self.vectorizer is None:
            from sklearn.feature_extraction.text import HashingVectorizer
            self.vectorizer = HashingVectorizer(
                n_features=self.settings.embedding_dimension,
                alternate_sign=True,
                norm=cast(Any, None),
                analyzer="word",
                ngram_range=(1, 2)
            )
        return self.vectorizer

    def _embed_text(self, text: str) -> np.ndarray:
        """Generates an L2-normalized dense embedding vector for transcript buffers."""
        if not text or not text.strip():
            return np.zeros(self.settings.embedding_dimension, dtype=np.float32)

        api_key = getattr(self.settings, "llm_api_key", "") or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "") or os.getenv("LLM_API_KEY", "")
        if api_key and api_key.strip():
            try:
                from google import genai
                client = genai.Client(api_key=api_key)
                response = client.models.embed_content(
                    model=self.settings.embedding_model,
                    contents=text
                )
                if hasattr(response, "embeddings") and response.embeddings:
                    vec = np.array(response.embeddings[0].values, dtype=np.float32)
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                    return vec
            except Exception:
                pass

        # Local deterministic fallback
        from sklearn.preprocessing import normalize
        vec_sparse = self._get_vectorizer().transform([text])
        vec_dense = vec_sparse.toarray().astype(np.float32)
        return normalize(vec_dense, norm="l2", axis=1).astype(np.float32)[0]

    async def evaluate_t0_stability(
        self,
        current_transcript_buffer: str,
        previous_transcript_buffer: str
    ) -> Tuple[bool, float]:
        """
        Tier 0 Gate: Evaluates whether the incoming transcript buffer is complete and stable
        enough to warrant invoking T1 LLM reasoning, or if the controller should WAIT.
        """
        start_time = time.perf_counter()
        curr_text = current_transcript_buffer.strip()
        prev_text = previous_transcript_buffer.strip()

        if not curr_text:
            return False, 0.0

        words = curr_text.split()
        token_count = len(words)

        # 1. Short conversational phrases (e.g. "hi", "thanks", "okay sounds good") are inherently complete
        short_complete_patterns = [
            r"^(?:(?:hi|hello|hey|greetings|good morning|good afternoon|good evening|bye|goodbye|thanks|thank you|ok|okay|sounds good|cool|great|perfect|got it|makes sense|understood|all good|all set)\s*[,.!?]?\s*)+$",
            r"^.*\b(that answers my question|that's all|no more questions|all set|have a great day)\b[.!?]*$"
        ]
        if any(re.match(p, curr_text, re.IGNORECASE) for p in short_complete_patterns):
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if self.telemetry:
                try:
                    self.telemetry.log_t0_gate("stream", 1.0, True, elapsed_ms)
                except Exception:
                    pass
            return True, 1.0

        # 2. Check minimum token count or terminal punctuation completion
        min_tokens = getattr(self.settings, "t0_min_chunk_tokens", 5)
        has_terminal_punct = curr_text.endswith((".", "?", "!"))
        if token_count < min_tokens and not has_terminal_punct:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if self.telemetry:
                try:
                    self.telemetry.log_t0_gate("stream", 0.0, False, elapsed_ms)
                except Exception:
                    pass
            return False, 0.0

        # 3. Check for dangling grammatical particles / trailing unfinished connectors
        dangling_connectors = {
            "the", "a", "an", "and", "or", "to", "for", "with", "is", "are",
            "about", "what", "how", "if", "when", "can", "could", "should",
            "my", "our", "in", "at", "of", "that", "this", "because"
        }
        last_word = re.sub(r"[^\w]", "", words[-1].lower())
        if curr_text.endswith("...") or (last_word in dangling_connectors and not has_terminal_punct):
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if self.telemetry:
                try:
                    self.telemetry.log_t0_gate("stream", 0.0, False, elapsed_ms)
                except Exception:
                    pass
            return False, 0.0

        # 4. Compute embedding similarity between previous buffer and current buffer
        if prev_text:
            vec_curr = self._embed_text(curr_text)
            vec_prev = self._embed_text(prev_text)
            similarity = float(np.dot(vec_curr, vec_prev))
        else:
            similarity = 1.0

        is_stable = has_terminal_punct or (token_count >= min_tokens)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if self.telemetry:
            try:
                self.telemetry.log_t0_gate("stream", similarity, is_stable, elapsed_ms)
            except Exception:
                pass

        return is_stable, similarity

    def _fallback_t1_reasoning(
        self,
        transcript: str,
        session_state: SessionState
    ) -> ControllerDecision:
        """
        Deterministic, rule-based decomposition and routing fallback when LLM API
        credentials are not configured or in offline testing environments.
        """
        clean_text = transcript.strip()

        # 1. Pure conversational or formatting checks (no retrieval needed)
        conversational_patterns = [
            r"^(?:(?:hi|hello|hey|greetings|good morning|good afternoon|good evening|bye|goodbye|thanks|thank you|ok|okay|sounds good|cool|great|perfect|got it|makes sense|understood|all good|all set)\s*[,.!?]?\s*)+$",
            r"^.*\b(that answers my question|that's all|no more questions|all set|have a great day)\b[.!?]*$"
        ]
        formatting_pattern = r"\b(summarize in bullets|format as table|make it shorter|simplify|translate to|reformat)\b"

        if any(re.match(pat, clean_text, re.IGNORECASE) for pat in conversational_patterns):
            return ControllerDecision(
                should_retrieve=False,
                is_refine=False,
                sub_queries=[],
                reasoning="Conversational greeting, acknowledgement, or gratitude. No corpus retrieval required."
            )

        if re.search(formatting_pattern, clean_text, re.IGNORECASE) and session_state.previous_answer:
            return ControllerDecision(
                should_retrieve=False,
                is_refine=False,
                sub_queries=[],
                reasoning="Formatting request on existing session state. No corpus retrieval required."
            )

        # 2. Refinement detection
        has_prior_context = bool(session_state.previous_answer or session_state.covered_intents or session_state.claims)
        refine_signals = [
            r"^(actually|instead|rather|what about|how about|also|and for|specifically for|now for|update|what if|in that case)\b",
            r"\b(instead|actually|what about|for international|for a group|for \d+ people)\b"
        ]
        is_refinement = has_prior_context and any(re.search(pat, clean_text, re.IGNORECASE) for pat in refine_signals)

        # 3. Multi-intent decomposition
        # Split on conjunctions / clauses that indicate multiple distinct queries
        split_pattern = r"\s+(?:and also|and what about|and what is|and what are|and how about|as well as|and)\s+"
        parts = re.split(split_pattern, clean_text, flags=re.IGNORECASE)
        
        # Clean and filter sub-queries
        sub_queries: List[str] = []
        for part in parts:
            part_clean = part.strip().rstrip("?.,!")
            # Remove leading conjunctions or conversational fillers
            part_clean = re.sub(r"^(also|what about|how about|and|can you tell me|what is|what are)\s+", "", part_clean, flags=re.IGNORECASE).strip()
            if part_clean and len(part_clean.split()) >= 2:
                # Add question framing if needed
                sub_queries.append(part.strip())

        if not sub_queries:
            sub_queries = [clean_text]

        # In refinement mode with prior context, focus sub-query on the delta
        if is_refinement:
            reasoning = f"Session refinement detected with prior context. Decomposed into {len(sub_queries)} delta sub-queries."
        else:
            reasoning = f"Corpus query identified. Decomposed into {len(sub_queries)} sub-queries."

        return ControllerDecision(
            should_retrieve=True,
            is_refine=is_refinement,
            sub_queries=sub_queries,
            reasoning=reasoning
        )

    async def evaluate_t1_decision(
        self,
        transcript: str,
        session_state: SessionState
    ) -> ControllerDecision:
        """
        Tier 1 Gate: Reads SessionState and performs structured routing and multi-intent
        decomposition (via Google GenAI or deterministic semantic fallback).
        """
        start_time = time.perf_counter()
        api_key = getattr(self.settings, "llm_api_key", "") or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "") or os.getenv("LLM_API_KEY", "")

        if api_key and api_key.strip():
            try:
                from google import genai
                from google.genai import types
                from pydantic import BaseModel, Field

                class T1StructuredDecision(BaseModel):
                    should_retrieve: bool = Field(description="True if factual policy evidence from the corpus is required, False for conversational or formatting turns.")
                    is_refine: bool = Field(description="True if this turn refines, clarifies, or updates an existing answer/session state.")
                    sub_queries: List[str] = Field(default_factory=list, description="Atomic search queries targeting missing intents.")
                    reasoning: str = Field(description="Brief explanation of the decision.")

                client = genai.Client(api_key=api_key)
                
                prompt = f"""You are the Tier 1 Routing & Multi-Intent Decomposition Controller for an Enterprise Live RAG system.
Analyze the user's transcript in the context of the active session state.

[Session State]
- Covered Intents: {session_state.covered_intents}
- Prior Claims: {[c.text for c in session_state.claims]}
- Prior Answer: {session_state.previous_answer}

[Incoming Transcript]
"{transcript}"

Decide:
1. should_retrieve: Is corpus search necessary? (False for greetings, formatting, acknowledgements).
2. is_refine: Does this update/refine prior context?
3. sub_queries: List of atomic, specific queries for retrieval. If multi-intent, decompose into separate queries. If refinement, search ONLY for the delta.
4. reasoning: Short explanation.
"""
                response = client.models.generate_content(
                    model=self.settings.llm_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=T1StructuredDecision,
                        temperature=self.settings.llm_temperature
                    )
                )

                if response.text:
                    parsed = json.loads(response.text)
                    decision = ControllerDecision(
                        should_retrieve=parsed.get("should_retrieve", True),
                        is_refine=parsed.get("is_refine", False),
                        sub_queries=parsed.get("sub_queries", [transcript]),
                        reasoning=parsed.get("reasoning", "")
                    )
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    if self.telemetry:
                        try:
                            self.telemetry.log_t1_decision(
                                session_id=session_state.session_id,
                                retrieve=decision.should_retrieve,
                                is_refine=decision.is_refine,
                                sub_queries=decision.sub_queries,
                                answer_version=session_state.answer_version,
                                latency_ms=elapsed_ms
                            )
                        except Exception:
                            pass
                    return decision
            except Exception as e:
                # Log warning and proceed to fallback
                pass

        # Fallback deterministic router
        decision = self._fallback_t1_reasoning(transcript, session_state)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if self.telemetry:
            try:
                self.telemetry.log_t1_decision(
                    session_id=session_state.session_id,
                    retrieve=decision.should_retrieve,
                    is_refine=decision.is_refine,
                    sub_queries=decision.sub_queries,
                    answer_version=session_state.answer_version,
                    latency_ms=elapsed_ms
                )
            except Exception:
                pass
        return decision

    async def process_incoming_transcript(
        self,
        current_transcript: str,
        previous_transcript: str,
        session_state: SessionState
    ) -> ControllerDecision:
        """
        Orchestrates end-to-end controller workflow:
        1. Evaluate T0 intent stability gate.
        2. If not stable -> return ControllerDecision with t0_stable=False, should_retrieve=False (WAIT).
        3. If stable -> evaluate T1 structured decision with SessionState awareness.
        """
        is_stable, similarity = await self.evaluate_t0_stability(
            current_transcript_buffer=current_transcript,
            previous_transcript_buffer=previous_transcript
        )

        if not is_stable:
            return ControllerDecision(
                should_retrieve=False,
                is_refine=False,
                sub_queries=[],
                reasoning=f"Utterance incomplete or actively drifting (similarity: {similarity:.2f}). Waiting for speech stabilization.",
                t0_stable=False
            )

        decision = await self.evaluate_t1_decision(
            transcript=current_transcript,
            session_state=session_state
        )
        decision.t0_stable = True
        return decision

