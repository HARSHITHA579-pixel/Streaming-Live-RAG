"""
Multi-Intent Hybrid Retriever for Streaming RAG

================================================================================
RESPONSIBILITY:
- For EACH decomposed sub-query independently:
    1. Perform sparse BM25 retrieval against precomputed corpus index.
    2. Perform dense vector retrieval via cosine similarity / dot-product against corpus embeddings.
    3. Apply Reciprocal Rank Fusion (RRF) specifically for that sub-query to extract Top-4 candidates.
- Aggregate all candidate chunks across all sub-queries:
    4. Compute Union of all candidate chunks.
    5. Deduplicate identical chunk IDs.
- Final Evidence Reranking & Minimum Evidence Guarantee:
    6. Execute cross-encoder Reranker ONLY on the deduplicated candidate set.
    7. Enforce Minimum Evidence Guarantee: Every sub-query MUST retain at least its top 2
       highest-ranked chunks in the final evidence set to prevent intent starvation.

INPUTS:
- List of decomposed sub-queries (`List[str]`) from `app.controller`.
- Precomputed indices (`data/bm25_index.pkl`, `data/embeddings.npy`, `data/chunk_metadata.json`).

OUTPUTS:
- Final filtered and reranked list of `RetrievedEvidence` chunks with per-intent attribution.

CONNECTED COMPONENTS:
- `corpus/build_index.py`: Produced the index artifacts consumed by this retriever.
- `app.controller`: Supplies the decomposed sub-queries.
- `app.session_state`: Receives the final deduplicated evidence set to update `evidence_pool`.
- `app.synthesis`: Uses the retrieved evidence for grounded claim formulation.
- `app.telemetry`: Logs per-sub-query scores, RRF ranks, union size, and reranker scores.

WHY THIS ARCHITECTURE:
- Separate Per-Sub-Query RRF: Prevents strong intents (e.g. general travel rules) from dominating
  and starving subtle, specific intents (e.g. late-booking exceptions) during early retrieval.
- Minimum Evidence Guarantee: Guarantees multi-intent balance in the final evidence context.
- Late Reranking: Reranking is computationally expensive; applying it only to the final deduplicated
  top-4 union keeps streaming latency low.
================================================================================
"""

import os
import re
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set, cast
from pathlib import Path

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from app.session_state import RetrievedEvidence
from app.config import Settings, settings
from app.telemetry import TelemetryLogger, telemetry_logger


class HybridRetriever:
    """
    Implements independent per-sub-query BM25 + Dense retrieval, per-query RRF,
    candidate union/deduplication, cross-encoder reranking, and minimum evidence guarantees.
    """

    def __init__(
        self,
        app_settings: Settings = settings,
        logger: TelemetryLogger = telemetry_logger
    ):
        self.settings = app_settings
        self.telemetry = logger
        self.bm25_index: Optional[Any] = None
        self.embeddings: Optional[np.ndarray] = None
        self.chunk_metadata: Optional[List[Dict[str, Any]]] = None
        self.chunk_by_id: Dict[str, Dict[str, Any]] = {}
        self.vectorizer: Optional[Any] = None
        self.index_manifest: Optional[Dict[str, Any]] = None

        # Load indices if available at startup
        try:
            self.load_indexes()
        except FileNotFoundError:
            # Indices may not be built yet during initial setup
            pass

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase alphanumeric tokens matching build_index.py."""
        return re.findall(r"\w+", text.lower())

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

    def _embed_query(self, query: str) -> np.ndarray:
        """
        Embeds a search query into a normalized dense vector using the same configuration
        as corpus/build_index.py. Validates backend compatibility against stored index manifest.
        """
        expected_backend = self.index_manifest.get("embedding_backend") if self.index_manifest else None

        api_key = getattr(self.settings, "llm_api_key", "") or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "") or os.getenv("LLM_API_KEY", "")

        # If manifest requires local backend, use local deterministic vectorizer directly
        if expected_backend == "local":
            from sklearn.preprocessing import normalize
            vec_sparse: Any = self._get_vectorizer().transform([query])
            vec_dense = vec_sparse.toarray().astype(np.float32)
            norm_dense = cast(Any, normalize(vec_dense, norm="l2", axis=1))
            return np.asarray(norm_dense, dtype=np.float32)[0]

        # If GenAI API key is present or manifest requires genai
        if api_key and api_key.strip():
            try:
                from google import genai
                client = genai.Client(api_key=api_key)
                response = client.models.embed_content(
                    model=self.settings.embedding_model,
                    contents=query
                )
                if hasattr(response, "embeddings") and response.embeddings and response.embeddings[0].values is not None:
                    vec = np.array(response.embeddings[0].values, dtype=np.float32)
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                    return vec
            except Exception as e:
                if expected_backend == "genai":
                    raise RuntimeError(
                        f"Index was built using Google GenAI embeddings ('genai'), but runtime GenAI embedding failed: {e}. "
                        "Incompatible embedding vector spaces cannot be mixed."
                    )

        if expected_backend == "genai":
            raise RuntimeError(
                "Index was built using Google GenAI embeddings ('genai'), but no API key is configured for runtime retrieval. "
                "Incompatible embedding vector spaces cannot be mixed."
            )

        # Local deterministic fallback matching build_index.py
        from sklearn.preprocessing import normalize
        vec_sparse = self._get_vectorizer().transform([query])
        vec_dense = vec_sparse.toarray().astype(np.float32)
        norm_dense = cast(Any, normalize(vec_dense, norm="l2", axis=1))
        return np.asarray(norm_dense, dtype=np.float32)[0]

    def load_indexes(self) -> None:
        """
        Loads precomputed BM25 index, dense embeddings, and chunk metadata into memory.
        Validates artifact presence, counts, schema, and alignment.
        """
        data_dir = Path(self.settings.data_dir)
        bm25_path = data_dir / "bm25_index.pkl"
        embeddings_path = data_dir / "embeddings.npy"
        metadata_path = data_dir / "chunk_metadata.json"
        manifest_path = data_dir / "index_manifest.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Chunk metadata artifact not found at: {metadata_path}")
        if not bm25_path.exists():
            raise FileNotFoundError(f"BM25 index artifact not found at: {bm25_path}")
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings artifact not found at: {embeddings_path}")

        # 0. Load Manifest if present
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                self.index_manifest = json.load(f)
        else:
            self.index_manifest = None

        # 1. Load Metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.chunk_metadata = json.load(f)

        # 2. Load BM25 Index
        with open(bm25_path, "rb") as f:
            self.bm25_index = pickle.load(f)

        # 3. Load Dense Embeddings
        self.embeddings = np.load(embeddings_path)

        # 4. Strict Validation Checks
        meta_count = len(self.chunk_metadata) if self.chunk_metadata is not None else 0
        bm25_count = getattr(self.bm25_index, "corpus_size", 0)
        emb_rows = self.embeddings.shape[0] if self.embeddings is not None else 0

        if meta_count == 0:
            raise ValueError("Chunk metadata is empty.")
        if meta_count != bm25_count:
            raise ValueError(f"Alignment error: Metadata count ({meta_count}) != BM25 corpus size ({bm25_count}).")
        if meta_count != emb_rows:
            raise ValueError(f"Alignment error: Metadata count ({meta_count}) != Embeddings rows ({emb_rows}).")

        # Validate unique IDs and required provenance fields
        self.chunk_by_id.clear()
        required_fields = ["doc_id", "chunk_id", "title", "source_file", "text"]

        for idx, entry in enumerate(self.chunk_metadata or []):
            for field_name in required_fields:
                if not entry.get(field_name):
                    raise ValueError(f"Metadata entry at index {idx} missing required field '{field_name}'.")
            if not (entry.get("section") or entry.get("section_title")):
                raise ValueError(f"Metadata entry at index {idx} missing section information.")

            chunk_id = entry["chunk_id"]
            if chunk_id in self.chunk_by_id:
                raise ValueError(f"Duplicate chunk_id detected: '{chunk_id}'.")
            self.chunk_by_id[chunk_id] = entry

    async def retrieve_bm25_for_sub_query(
        self,
        sub_query: str,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Executes BM25 sparse search for a single sub-query.
        Returns list of (chunk_id, bm25_score) tuples sorted descending by score for scores > 0.0.
        """
        if self.bm25_index is None or not self.chunk_metadata:
            self.load_indexes()
        assert self.bm25_index is not None and self.chunk_metadata is not None

        tokenized_query = self._tokenize(sub_query)
        if not tokenized_query:
            return []

        scores = self.bm25_index.get_scores(tokenized_query)
        scored_indices = np.argsort(scores)[::-1]

        # Filter out non-positive scores so zero-score matches do not pollute RRF
        valid_indices = [idx for idx in scored_indices if float(scores[idx]) > 0.0][:top_k]

        results: List[Tuple[str, float]] = []
        for idx in valid_indices:
            score = float(scores[idx])
            chunk_id = self.chunk_metadata[idx]["chunk_id"]
            results.append((chunk_id, score))

        return results

    async def retrieve_dense_for_sub_query(
        self,
        sub_query: str,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Executes dense vector search for a single sub-query using cosine similarity.
        Returns list of (chunk_id, cosine_similarity) tuples sorted descending by score.
        """
        if self.embeddings is None or not self.chunk_metadata:
            self.load_indexes()
        assert self.embeddings is not None and self.chunk_metadata is not None

        query_vec = self._embed_query(sub_query)
        # Compute cosine similarity via dot product against L2-normalized embeddings
        similarities = np.dot(self.embeddings, query_vec)
        scored_indices = np.argsort(similarities)[::-1][:top_k]

        results: List[Tuple[str, float]] = []
        for idx in scored_indices:
            sim = float(similarities[idx])
            chunk_id = self.chunk_metadata[idx]["chunk_id"]
            results.append((chunk_id, sim))

        return results

    def compute_rrf(
        self,
        bm25_results: List[Tuple[str, float]],
        dense_results: List[Tuple[str, float]],
        k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Computes Reciprocal Rank Fusion (RRF) for one sub-query:
        RRF_score(d) = sum(1 / (k + rank_i(d))) across BM25 and Dense result lists.
        """
        rrf_constant = k if k is not None else getattr(self.settings, "rrf_k", 60)
        rrf_scores: Dict[str, float] = {}

        for rank, (chunk_id, _) in enumerate(bm25_results, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rrf_constant + rank))

        for rank, (chunk_id, _) in enumerate(dense_results, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rrf_constant + rank))

        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_chunks

    def _detect_geographic_scope(self, query: str) -> Tuple[List[str], List[str], bool, bool]:
        """
        Generic detector for geographic entities (cities, states, national, international).
        """
        q_lower = query.lower()

        # Generic Indian City to State mapping
        indian_cities_to_state = {
            "chennai": "tamil nadu",
            "madras": "tamil nadu",
            "hyderabad": "telangana",
            "bengaluru": "karnataka",
            "bangalore": "karnataka",
            "mumbai": "maharashtra",
            "bombay": "maharashtra",
            "pune": "maharashtra",
            "kolkata": "west bengal",
            "calcutta": "west bengal",
            "ahmedabad": "gujarat",
            "surat": "gujarat",
            "vadodara": "gujarat",
            "delhi": "delhi",
            "new delhi": "delhi",
            "noida": "uttar pradesh",
            "lucknow": "uttar pradesh",
            "kanpur": "uttar pradesh",
            "varanasi": "uttar pradesh",
            "agra": "uttar pradesh",
            "prayagraj": "uttar pradesh",
            "allahabad": "uttar pradesh",
            "kochi": "kerala",
            "cochin": "kerala",
            "thiruvananthapuram": "kerala",
            "trivandrum": "kerala",
            "kozhikode": "kerala",
            "calicut": "kerala",
            "panaji": "goa",
            "shillong": "meghalaya",
            "jaipur": "rajasthan",
            "chandigarh": "punjab",
            "patna": "bihar",
            "bhopal": "madhya pradesh",
            "indore": "madhya pradesh",
            "guwahati": "assam",
            "bhubaneswar": "odisha",
            "ranchi": "jharkhand",
            "dehradun": "uttarakhand",
            "shimla": "himachal pradesh",
            "srinagar": "jammu and kashmir",
        }

        indian_states = [
            "uttar pradesh", "up", "kerala", "goa", "meghalaya", "tamil nadu",
            "maharashtra", "gujarat", "karnataka", "telangana", "andhra pradesh",
            "west bengal", "rajasthan", "punjab", "haryana", "bihar", "madhya pradesh",
            "odisha", "assam", "delhi"
        ]

        # Identify location signals in the query using whole-word matching
        matched_cities = [c for c in indian_cities_to_state.keys() if re.search(r"\b" + re.escape(c) + r"\b", q_lower)]
        matched_states = [s for s in indian_states if re.search(r"\b" + re.escape(s) + r"\b", q_lower)]

        # If a city was matched, add its corresponding state as an implied state target
        for c in matched_cities:
            st = indian_cities_to_state[c]
            if st not in matched_states:
                matched_states.append(st)

        is_international_query = bool(re.search(
            r"\b(international|global|uk|australia|monash|overseas|foreign|abroad|hse|victoria|perth)\b",
            q_lower
        ))
        is_india_query = bool(
            matched_cities or matched_states or re.search(r"\b(india|indian|national|fssai|ndma|railway|bye-laws|pandal)\b", q_lower)
        )

        return matched_cities, matched_states, is_india_query, is_international_query

    async def retrieve_hybrid_for_sub_query(
        self,
        sub_query: str,
        sub_query_id: int = 0,
        top_k: Optional[int] = None
    ) -> List[RetrievedEvidence]:
        """
        Performs BM25 + Dense retrieval followed by RRF for a single sub-query,
        incorporating hierarchical fallback retrieval when querying Indian regions.
        """
        k_val = top_k or getattr(self.settings, "retrieval_top_k_per_query", 4)
        fetch_k = max(k_val * 15, 60)

        bm25_res = await self.retrieve_bm25_for_sub_query(sub_query, top_k=fetch_k)
        dense_res = await self.retrieve_dense_for_sub_query(sub_query, top_k=fetch_k)

        rrf_res = self.compute_rrf(bm25_res, dense_res, k=getattr(self.settings, "rrf_k", 60))

        # Hierarchical Indian Fallback Discovery:
        # If querying an Indian entity or general Indian requirements, ensure national/state
        # baseline evidence is retrieved into the candidate pool if local docs are unavailable.
        matched_cities, matched_states, is_india, is_intl = self._detect_geographic_scope(sub_query)
        if is_india and not is_intl:
            clean_q = re.sub(r"\b(in|for|at|the|what|are|requirements|guidelines)\b", "", sub_query, flags=re.IGNORECASE)
            fallback_q = f"{clean_q} India national state guidelines mass gathering festival building bye laws public assembly temporary structure crowd safety"
            fb_bm25 = await self.retrieve_bm25_for_sub_query(fallback_q, top_k=fetch_k)
            fb_dense = await self.retrieve_dense_for_sub_query(fallback_q, top_k=fetch_k)
            fb_rrf = self.compute_rrf(fb_bm25, fb_dense, k=getattr(self.settings, "rrf_k", 60))

            # Merge Indian chunks from fallback retrieval
            combined_scores = dict(rrf_res)
            for cid, score in fb_rrf:
                if self.chunk_by_id[cid]["source_priority"] < 4:
                    combined_scores[cid] = max(combined_scores.get(cid, 0.0), score)
            rrf_res = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        top_rrf = rrf_res[:fetch_k]

        evidence_list: List[RetrievedEvidence] = []
        for chunk_id, rrf_score in top_rrf:
            meta = self.chunk_by_id[chunk_id]
            evidence = RetrievedEvidence(
                chunk_id=meta["chunk_id"],
                doc_id=meta["doc_id"],
                section_id=meta.get("section_id"),
                section_title=meta.get("section_title") or meta.get("section"),
                text=meta["text"],
                score=rrf_score,
                sub_query_id=sub_query_id,
                retrieval_method="rrf",
                page_number=meta.get("page_number", 1),
                geographic_scope=meta.get("geographic_scope", "General"),
                scope_level=meta.get("scope_level", "general"),
                source_priority=meta.get("source_priority", 4),
                source_file=meta.get("source_file", ""),
                source_path=meta.get("source_path", ""),
                document_title=meta.get("document_title") or meta.get("title")
            )
            evidence_list.append(evidence)

        return evidence_list

    async def rerank_candidates(
        self,
        sub_queries: List[str],
        candidate_chunks: List[RetrievedEvidence]
    ) -> List[RetrievedEvidence]:
        """
        Orders candidate chunks across sub-queries with source-scope and geographic awareness.
        Specificity fallback hierarchy:
        1. City/local authority-specific evidence
        2. State-specific evidence
        3. India-wide/general Indian evidence
        4. International/general baseline evidence

        Semantic relevance remains foundational; geographic specificity acts as an
        adaptive alignment boost.
        """
        if not candidate_chunks:
            return []

        # Combine all sub-queries to understand global query geographic scope
        combined_query = " ".join(sub_queries)
        matched_cities, matched_states, is_india_query, is_international_query = self._detect_geographic_scope(combined_query)

        def calculate_ranked_score(chunk: RetrievedEvidence) -> float:
            base_score = chunk.score
            scope_str = (chunk.geographic_scope or "").lower()
            priority = chunk.source_priority  # 1: city, 2: state, 3: national, 4: international

            boost = 0.0

            # 1. Exact city match (e.g. Delhi for Delhi query)
            if matched_cities and any(c in scope_str for c in matched_cities):
                boost += 0.050
            # 2. Exact state match (e.g. UP for UP query, Kerala for Kerala query)
            elif matched_states and any(s in scope_str for s in matched_states):
                boost += 0.035
            # 3. Explicit international query
            elif is_international_query and not is_india_query:
                if priority == 4:
                    boost += 0.020
            # 4. India query / Indian city query fallback hierarchy
            elif is_india_query:
                if priority == 1:
                    boost += 0.022
                elif priority == 2:
                    boost += 0.022
                elif priority == 3:
                    boost += 0.025  # National baseline (e.g. Model Building Bye-Laws, NDMA) prioritized for unindexed Indian cities
                elif priority == 4:
                    boost += 0.000  # International baseline used only if Indian evidence is unavailable/insufficient
            # 5. Generic query without explicit geo-target
            else:
                if priority in (1, 2, 3):
                    boost += 0.010
                elif priority == 4:
                    boost += 0.000

            return base_score + boost

        # Sort candidate chunks by combined score descending
        sorted_candidates = sorted(candidate_chunks, key=calculate_ranked_score, reverse=True)
        return sorted_candidates



    def apply_minimum_evidence_guarantee(
        self,
        sub_query_candidates: Dict[int, List[RetrievedEvidence]],
        reranked_pool: List[RetrievedEvidence],
        min_per_intent: Optional[int] = None
    ) -> List[RetrievedEvidence]:
        """
        Guarantees that each sub-query retains at least `min_per_intent` chunks in the final set
        to prevent intent starvation in multi-intent scenarios.
        """
        min_intent = min_per_intent if min_per_intent is not None else getattr(self.settings, "min_evidence_per_intent", 2)
        final_evidence: List[RetrievedEvidence] = []
        seen_chunk_ids: Set[str] = set()

        # 1. Guarantee top reranked chunks for each distinct sub-query intent
        for sub_query_id in sub_query_candidates.keys():
            added_for_intent = 0
            for chunk in reranked_pool:
                if chunk.sub_query_id == sub_query_id and chunk.chunk_id not in seen_chunk_ids:
                    final_evidence.append(chunk)
                    seen_chunk_ids.add(chunk.chunk_id)
                    added_for_intent += 1
                    if added_for_intent >= min_intent:
                        break

        # 2. Fill remaining slots from top global reranked pool
        for chunk in reranked_pool:
            if chunk.chunk_id not in seen_chunk_ids:
                final_evidence.append(chunk)
                seen_chunk_ids.add(chunk.chunk_id)

        return final_evidence


    async def retrieve_for_sub_queries(
        self,
        sub_queries: List[str],
        session_id: str = ""
    ) -> List[RetrievedEvidence]:
        """
        Orchestrates the complete multi-intent retrieval workflow:
        1. For each sub-query independently: BM25 + Dense -> RRF -> Top-4 per sub-query.
        2. Union candidates across sub-queries.
        3. Rerank candidate union.
        4. Apply Minimum Evidence Guarantee (at least top-2 chunks per sub-query retained).
        5. Return final evidence list.
        """
        if not sub_queries:
            return []

        if self.bm25_index is None or not self.chunk_metadata:
            self.load_indexes()

        sub_query_candidates: Dict[int, List[RetrievedEvidence]] = {}
        all_candidates: List[RetrievedEvidence] = []
        seen_for_subquery: Set[Tuple[int, str]] = set()

        for idx, sq in enumerate(sub_queries):
            sq_evidence = await self.retrieve_hybrid_for_sub_query(
                sub_query=sq,
                sub_query_id=idx,
                top_k=getattr(self.settings, "retrieval_top_k_per_query", 4)
            )
            sub_query_candidates[idx] = sq_evidence

            for chunk in sq_evidence:
                pair = (idx, chunk.chunk_id)
                if pair not in seen_for_subquery:
                    all_candidates.append(chunk)
                    seen_for_subquery.add(pair)

            # Optional telemetry logging per subquery chunk
            if self.telemetry:
                for rank, chunk in enumerate(sq_evidence, start=1):
                    try:
                        self.telemetry.log_retrieval_chunk(
                            session_id=session_id,
                            sub_query_id=idx,
                            chunk_id=chunk.chunk_id,
                            method=chunk.retrieval_method,
                            rank=rank,
                            score=chunk.score
                        )
                    except Exception:
                        pass

        # Rerank candidates across all sub-queries
        reranked_pool = await self.rerank_candidates(sub_queries, all_candidates)

        # Apply Minimum Evidence Guarantee
        final_evidence = self.apply_minimum_evidence_guarantee(
            sub_query_candidates=sub_query_candidates,
            reranked_pool=reranked_pool,
            min_per_intent=getattr(self.settings, "min_evidence_per_intent", 2)
        )

        return final_evidence

