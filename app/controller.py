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
    refinement_type: str = "none"  # "supersession", "continuation", "removal", "additive", "none"
    intent_type: str = "REQUIREMENTS"  # INFORMATION_LOOKUP, REQUIREMENTS, COMPARISON, RECOMMENDATION, LOCATION_VENUE_REQUEST, BOOKING_ACTION, REFINEMENT, CONVERSATIONAL, UNSUPPORTED
    is_supersession: bool = False
    is_removal: bool = False
    is_continuation: bool = False
    superseded_terms: List[str] = field(default_factory=list)

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
        norm_dense = cast(Any, normalize(vec_dense, norm="l2", axis=1))
        return np.asarray(norm_dense, dtype=np.float32)[0]

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
        has_terminal_punct = curr_text.endswith((".", "?", "!"))

        # 1. Short conversational phrases (e.g. "hi", "thanks", "okay", "yes", "hmm") are inherently complete
        short_complete_patterns = [
            r"^(?:(?:hi|hello|hey|greetings|good morning|good afternoon|good evening|bye|goodbye|thanks|thank you|ok|okay|sounds good|cool|great|perfect|got it|makes sense|understood|all good|all set|hmm|yes|yeah|sure|no|yep)\s*(?:nexa)?\s*[,.!?]?\s*)+$",
            r"^.*\b(that answers my question|that's all|no more questions|all set|have a great day)\b[.!?]*$"
        ]
        if any(re.match(p, curr_text, re.IGNORECASE) for p in short_complete_patterns):
            return True, 1.0

        # 2. Check for dangling grammatical particles / trailing unfinished connectors
        dangling_connectors = {
            "the", "a", "an", "and", "or", "to", "for", "with", "is", "are",
            "about", "what", "how", "if", "when", "can", "could", "should",
            "my", "our", "in", "at", "of", "that", "this", "because", "as", "also", "actually"
        }
        last_word = re.sub(r"[^\w]", "", words[-1].lower())
        if curr_text.endswith("...") or (last_word in dangling_connectors and not has_terminal_punct):
            return False, 0.0

        # 3. Check minimum token count or terminal punctuation completion
        min_tokens = getattr(self.settings, "t0_min_chunk_tokens", 5)
        is_continuation_or_refine = bool(re.search(r"^(?:for|actually|instead|rather|forget|only|keep|drop|remove|with|in)\b", curr_text, re.IGNORECASE))

        if token_count < 2 and not has_terminal_punct:
            return False, 0.0

        # 4. Compute embedding similarity between previous buffer and current buffer
        if prev_text:
            vec_curr = self._embed_text(curr_text)
            vec_prev = self._embed_text(prev_text)
            similarity = float(np.dot(vec_curr, vec_prev))
        else:
            similarity = 1.0

        is_stable = has_terminal_punct or (token_count >= min_tokens) or is_continuation_or_refine or (token_count >= 2 and last_word not in dangling_connectors)

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
            r"^(?:(?:hi|hello|hey|greetings|good morning|good afternoon|good evening|bye|goodbye|thanks|thank you|ok|okay|sounds good|cool|great|perfect|got it|makes sense|understood|all good|all set|hmm|yes|yeah|sure|no|yep)\s*(?:nexa)?\s*[,.!?]?\s*)+$",
            r"^.*\b(that answers my question|that's all|no more questions|all set|have a great day)\b[.!?]*$"
        ]
        formatting_pattern = r"\b(summarize in bullets|format as table|format this|make it shorter|simplify|translate to|reformat|repeat (?:your )?(?:last )?answer|in bullet points|bullet points|as bullets)\b"

        if any(re.match(pat, clean_text, re.IGNORECASE) for pat in conversational_patterns):
            return ControllerDecision(
                should_retrieve=False,
                is_refine=False,
                intent_type="CONVERSATIONAL",
                sub_queries=[],
                reasoning="Conversational greeting, acknowledgement, or gratitude. No corpus retrieval required."
            )

        if re.search(formatting_pattern, clean_text, re.IGNORECASE):
            return ControllerDecision(
                should_retrieve=False,
                is_refine=False,
                intent_type="FORMATTING",
                sub_queries=[],
                reasoning="Formatting request. No corpus retrieval required."
            )

        # Detect commercial venue, booking, vendor, phone number, and out-of-corpus directory requests
        is_policy_inquiry = bool(re.search(r"\b(reimbursement|per diem|rate limit|policy|policies|rules for|guidelines for|regulations)\b", clean_text, re.IGNORECASE))

        venue_patterns = [
            r"\b(dining hall|banquet hall|wedding venue|party hall|buffet)\b",
            r"\b(find (?:me )?(?:a )?(?:hotel|restaurant|venue|dining hall|banquet|hall|place))\b",
            r"\b(need (?:a )?(?:dining hall|banquet hall|hotel|restaurant|wedding venue|venue|place to eat))\b",
            r"\b(recommend (?:a )?(?:restaurant|hotel|venue|dining hall|place to eat|buffet))\b",
            r"\b(best restaurant|best hotel|best buffet|cheapest hotel|cheapest venue|top restaurant)\b",
            r"\b(which (?:hotel|restaurant|venue|cafe|buffet))\b",
            r"\b(hotel in [A-Za-z]+|restaurant in [A-Za-z]+|venue in [A-Za-z]+|restaurant near [A-Za-z]+|hotel near [A-Za-z]+|venue near [A-Za-z]+)\b",
            r"\b(where can i (?:rent|find|book|hire) (?:a |me )?(?:hall|venue|hotel|restaurant|room|chairs?|tents?))\b",
            r"\b(hall for \d+|venue for \d+|hotel for \d+|dining hall for \d+|room for rent|rent a hall|rent a venue)\b"
        ]
        is_venue_request = not is_policy_inquiry and any(re.search(pat, clean_text, re.IGNORECASE) for pat in venue_patterns)

        booking_action_patterns = [
            r"\b(book (?:a |me )?(?:banquet|hall|hotel|table|flight|ticket|venue|seats?|room))\b",
            r"\b(reserve (?:a |me )?(?:table|seats?|hotel|room|hall|venue))\b",
            r"\b(order \d+ (?:box|lunches|meals|pizzas?))\b",
            r"\b(rent chairs|hire event|hire vendor|flight tickets?|book taxi|rental vendor)\b"
        ]
        is_booking_action = not is_policy_inquiry and any(re.search(pat, clean_text, re.IGNORECASE) for pat in booking_action_patterns)

        unsupported_directory_patterns = [
            r"\b(phone number|contact number|mobile number|email address|price to rent|how much does it cost|weather forecast|weather tomorrow|weather for|3-course dinner|dinner menu|menu for|vendor near me|rental vendor|sound system)\b",
            r"\b(give me the (?:phone|contact) number)\b",
            r"\b(what is the (?:price|cost|weather|phone|menu))\b"
        ]
        is_unsupported_directory = any(re.search(pat, clean_text, re.IGNORECASE) for pat in unsupported_directory_patterns)

        has_prior_context = bool(session_state.previous_answer or session_state.active_intents or session_state.covered_intents or session_state.claims)

        # 2. Same-Turn Refinement / Correction Detection (e.g., "Pune travel, actually international travel")
        same_turn_split = re.split(r",?\s+(?:actually|no\s+wait|rather|scratch\s+that|i\s+meant|actually\s+i\s+mean)\s+", clean_text, flags=re.IGNORECASE)
        if len(same_turn_split) >= 2:
            superseded_part = same_turn_split[0].strip()
            active_part = same_turn_split[-1].strip()
            # Clean sub-queries for the refined active part
            sub_q = re.sub(r"^(also|and|can you tell me|what is|what are|i mean|i meant)\s+", "", active_part, flags=re.IGNORECASE).strip()
            return ControllerDecision(
                should_retrieve=True,
                is_refine=True,
                intent_type="REFINEMENT",
                is_supersession=True,
                refinement_type="supersession",
                superseded_terms=[superseded_part],
                sub_queries=[sub_q or active_part],
                reasoning=f"Same-turn correction detected. Superseded '{superseded_part}', active target is '{active_part}'."
            )

        # 3. Explicit Removal Detection (e.g., "Forget hotel reimbursement, only cancellation")
        removal_match = re.search(r"^(?:forget|remove|drop|exclude|skip|omit|without|don't need|no longer need)\s+(?:about\s+)?(.+?)(?:,\s*(?:only|just|keep)\s+(.+))?$", clean_text, re.IGNORECASE)
        if removal_match:
            removed_term = removal_match.group(1).strip()
            retained_term = removal_match.group(2).strip() if removal_match.group(2) else ""
            sub_queries = [retained_term] if retained_term else []
            return ControllerDecision(
                should_retrieve=bool(sub_queries),
                is_refine=True,
                intent_type="REFINEMENT",
                is_removal=True,
                refinement_type="removal",
                superseded_terms=[removed_term],
                sub_queries=sub_queries,
                reasoning=f"Explicit removal detected. Removing intent '{removed_term}', retaining '{retained_term}'."
            )

        # 4. Cross-Turn Supersession / Correction Detection
        supersede_signals = [
            r"^(actually|instead|rather|no,|cancel that|wait,|scratch that|ignore that|forget|correct that|correction|not that)\b",
            r"\b(instead of|rather than|actually i mean|actually mean|i meant|i mean|actually for|actually what about|actually check)\b",
            r"^actually\b"
        ]
        is_supersession = has_prior_context and any(re.search(pat, clean_text, re.IGNORECASE) for pat in supersede_signals)

        if is_supersession:
            cleaned_query = re.sub(r"^(actually|instead|rather|no|wait|cancel that|scratch that|i meant|i mean|actually i mean|actually mean|actually for|actually what about|actually check)[,\s:]*", "", clean_text, flags=re.IGNORECASE).strip()
            cleaned_query = re.sub(r"^(i mean|i meant|for|about)\s+", "", cleaned_query, flags=re.IGNORECASE).strip()
            sub_queries = [cleaned_query or clean_text]
            return ControllerDecision(
                should_retrieve=True,
                is_refine=True,
                intent_type="REFINEMENT",
                is_supersession=True,
                refinement_type="supersession",
                sub_queries=sub_queries,
                reasoning="Cross-turn supersession detected. Superseding prior active context with refined query."
            )

        # 5. Continuation / Enrichment Detection (e.g. "for 30 people", "for a group", "in London")
        continuation_signals = [
            r"^(for|with|in|under|at|using|and for|specifically for|now for|what if|for a group|for \d+ people)\b"
        ]
        is_continuation = has_prior_context and any(re.search(pat, clean_text, re.IGNORECASE) for pat in continuation_signals)

        if is_continuation:
            prior_intent = session_state.active_intents[0] if session_state.active_intents else (session_state.covered_intents[0] if session_state.covered_intents else "")
            enriched_query = f"{prior_intent} {clean_text}".strip()
            return ControllerDecision(
                should_retrieve=True,
                is_refine=True,
                intent_type="REFINEMENT",
                is_continuation=True,
                refinement_type="continuation",
                sub_queries=[enriched_query, clean_text],
                reasoning=f"Continuation detected. Enriching active intent '{prior_intent}' with constraint '{clean_text}'."
            )

        # 6. Travel Reimbursement Refinement / Constraint Modifier Detection
        prior_context_list = session_state.active_intents + session_state.covered_intents + session_state.transcript_history
        is_travel_domain = has_prior_context and any(
            re.search(r"\b(travel|reimbursement|trip|per diem|employee trip)\b", t, re.IGNORECASE)
            for t in prior_context_list
        )

        if is_travel_domain:
            is_travel_modifier = bool(
                re.search(r"\b(trip\s+was|booking\s+was|trip\s+is|booking\s+is|after\s+travel|booked\s+after|before\s+travel|booking\s+timing)\b", clean_text, re.IGNORECASE)
                or (re.search(r"\b(international|domestic)\b", clean_text, re.IGNORECASE) and not is_venue_request and not is_booking_action)
            )
            if is_travel_modifier:
                sub_queries = []
                lower_text = clean_text.lower()
                if "international" in lower_text:
                    sub_queries.append("international employee travel reimbursement")
                elif "domestic" in lower_text:
                    sub_queries.append("domestic employee travel reimbursement")

                if any(k in lower_text for k in ["after travel", "after the trip", "booked after", "booking was made after", "booking made after", "booking after travel", "booking timing"]):
                    sub_queries.append("travel reimbursement booking made after travel")
                elif "booking" in lower_text or "reservation" in lower_text:
                    sub_queries.append("employee travel reimbursement booking timing and procedures")

                if not sub_queries:
                    sub_queries = [f"employee travel reimbursement {clean_text}"]

                return ControllerDecision(
                    should_retrieve=True,
                    is_refine=True,
                    intent_type="REFINEMENT",
                    refinement_type="additive",
                    is_continuation=True,
                    sub_queries=sub_queries,
                    reasoning="Travel reimbursement refinement detected. Preserving active domain 'travel_reimbursement' with constraints: " +
                              ("geographic_scope=international, " if 'international' in lower_text else "") +
                              ("booking_timing=after_travel" if any(k in lower_text for k in ['after travel', 'after the trip', 'booked after', 'booking was made after', 'booking made after']) else "")
                )

        # 7. Multi-Intent Decomposition (Additive)
        split_pattern = r"(?:,\s*|\s+(?:and also|and what about|and what is|and what are|and how about|as well as|and)\s+)"
        parts = re.split(split_pattern, clean_text, flags=re.IGNORECASE)
        
        sub_queries = []
        for part in parts:
            part_clean = part.strip().rstrip("?.,!")
            part_clean = re.sub(r"^(also|what about|how about|and|can you tell me|what is|what are|tell me about)\s+", "", part_clean, flags=re.IGNORECASE).strip()
            if part_clean and len(part_clean.split()) >= 1:
                sub_queries.append(part.strip())

        if not sub_queries:
            sub_queries = [clean_text]

        # Determine classified intent type
        if is_venue_request:
            inferred_intent = "LOCATION_VENUE_REQUEST"
        elif is_booking_action:
            inferred_intent = "BOOKING_ACTION"
        elif is_unsupported_directory:
            inferred_intent = "UNSUPPORTED"
        elif any(w in clean_text.lower() for w in ["compare", "difference between", "versus", "vs"]):
            inferred_intent = "COMPARISON"
        elif any(w in clean_text.lower() for w in ["recommend", "suggestion", "best practices"]):
            inferred_intent = "RECOMMENDATION"
        elif any(w in clean_text.lower() for w in ["requirement", "rules", "regulations", "standards", "safety", "measures"]):
            inferred_intent = "REQUIREMENTS"
        else:
            inferred_intent = "INFORMATION_LOOKUP"

        return ControllerDecision(
            should_retrieve=True,
            is_refine=False,
            intent_type=inferred_intent,
            refinement_type="additive" if len(sub_queries) > 1 else "none",
            sub_queries=sub_queries,
            reasoning=f"Corpus query identified as {inferred_intent}. Decomposed into {len(sub_queries)} sub-queries."
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
                    intent_type: str = Field(default="REQUIREMENTS", description="Intent type: INFORMATION_LOOKUP, REQUIREMENTS, COMPARISON, RECOMMENDATION, LOCATION_VENUE_REQUEST, BOOKING_ACTION, REFINEMENT, CONVERSATIONAL, UNSUPPORTED.")
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
2. is_refine: Does this update/refine prior context? (True if modifying, adding constraints, or clarifying previous turn).
3. intent_type: One of [INFORMATION_LOOKUP, REQUIREMENTS, COMPARISON, RECOMMENDATION, LOCATION_VENUE_REQUEST, BOOKING_ACTION, REFINEMENT, CONVERSATIONAL, UNSUPPORTED]. (Note: Our corpus contains event safety, crowd, food hygiene, accessibility, emergency standards, and employee travel reimbursement; it does NOT contain commercial hotel directories or booking systems).
4. sub_queries: List of atomic, specific queries for retrieval. If refinement/follow-up modifying an existing active domain (e.g. travel reimbursement), retain the active domain in each sub-query (e.g. for travel reimbursement with follow-up 'international and booking after travel', output ['international employee travel reimbursement', 'travel reimbursement booking made after travel']). Never search generic modifiers like 'international' alone.
5. reasoning: Short explanation.
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
                        intent_type=parsed.get("intent_type", "REQUIREMENTS"),
                        sub_queries=parsed.get("sub_queries", [transcript]),
                        reasoning=parsed.get("reasoning", "")
                    )
                    return decision
            except Exception as e:
                # Log warning and proceed to fallback
                pass

        # Fallback deterministic router
        decision = self._fallback_t1_reasoning(transcript, session_state)
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

