"""
Corpus Index Builder for Streaming RAG

================================================================================
RESPONSIBILITY:
- Read raw documents from `corpus/raw_docs/`.
- Discover and filter ONLY active corpus directories:
    01_event_planning, 02_food_catering, 03_travel_reimbursement,
    04_pune_venues, 05_india_general, 06_international_general
- Exclude old/inactive folders (02_crowd_management, 03_safety_emergency,
  04_event_permits_compliance, 05_accessibility) and deprecated files.
- Extract text page-by-page preserving document IDs, hierarchy, and section information.
- Chunk documents using configured chunk size and overlap strategies.
- Store granular metadata for every chunk (chunk_id, doc_id, category, city,
  source_file, section, content, token_count, geographic_scope).
- Build the sparse BM25 index and serialize it to `data/bm25_index.pkl`.
- Generate dense vector embeddings for all chunks and serialize to `data/embeddings.npy`.
- Save chunk metadata to `data/chunk_metadata.json` and index manifest to `data/index_manifest.json`.

INPUTS:
- Raw PDF documents in active folders under `corpus/raw_docs/`.
- Indexing hyperparameters from `app.config.Settings` (chunk_size, chunk_overlap, embedding model).

OUTPUTS:
- `data/bm25_index.pkl`: Serialized BM25 index.
- `data/embeddings.npy`: Dense embedding vectors (N x D float32 matrix).
- `data/chunk_metadata.json`: Chunk provenance records with category, city, and source_file.
- `data/index_manifest.json`: Manifest recording embedding backend, dimensions, and chunk count.

CONNECTED COMPONENTS:
- `app.config`: Reads indexing parameters (chunk sizes, embedding models, output paths).
- `app.retriever`: Consumes the generated BM25 index, embedding matrix, and chunk metadata.
================================================================================
"""

import os
import re
import json
import pickle
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, Set, cast
from pathlib import Path
import numpy as np

try:
    import pymupdf  # type: ignore
except ImportError:
    try:
        import fitz as pymupdf  # type: ignore
    except ImportError:
        pymupdf = None

try:
    import pypdf  # type: ignore
except ImportError:
    pypdf = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    from app.config import settings, Settings
except ImportError:
    @dataclass
    class SettingsFallback:
        chunk_size: int = 512
        chunk_overlap: int = 64
        embedding_model: str = "text-embedding-004"
        embedding_dimension: int = 768
        corpus_dir: Path = Path("corpus/raw_docs")
        data_dir: Path = Path("data")
        llm_api_key: str = ""
    settings = SettingsFallback()


# Active category normalization mapping
CATEGORY_ALIAS_MAP: Dict[str, str] = {
    "01_event_planning": "event_planning",
    "event_planning": "event_planning",
    "02_food_catering": "catering",
    "06_food_catering": "catering",
    "food_catering": "catering",
    "catering": "catering",
    "03_travel_reimbursement": "travel_reimbursement",
    "09_travel_reimbursement": "travel_reimbursement",
    "travel_reimbursement": "travel_reimbursement",
    "04_pune_venues": "pune_venue",
    "10_pune_venues": "pune_venue",
    "pune_venues": "pune_venue",
    "pune_venue": "pune_venue",
    "05_india_general": "india_general",
    "07_india_general": "india_general",
    "india_general": "india_general",
    "06_international_general": "international_general",
    "08_international_general": "international_general",
    "international_general": "international_general",
}

# Explicitly excluded old folders
EXCLUDED_FOLDERS: Set[str] = {
    "02_crowd_management",
    "crowd_management",
    "03_safety_emergency",
    "safety_emergency",
    "04_event_permits_compliance",
    "event_permits_compliance",
    "05_accessibility",
    "accessibility",
}

# Explicitly excluded deprecated files
EXCLUDED_FILENAMES: Set[str] = {
    "MODEL-BUILDING-BYE-LAWS-2016.pdf",
    "MODEL-BUILDING-BYE-LAWS.pdf",
    "model-building-bye-laws-2016.pdf",
    "model_building_bye_laws_2016.pdf",
    "model_building_bye_laws.pdf",
}

# Detailed document metadata registry for active corpus PDFs
DOCUMENT_REGISTRY: Dict[str, Dict[str, Any]] = {
    # 01_event_planning
    "14293-Events-Guidelines-2022.pdf": {
        "doc_id": "wa_events_guidelines_2022",
        "title": "Guidelines for Events in Western Australia 2022",
        "geographic_scope": "International",
        "scope_level": "international",
        "source_priority": 4,
        "category": "event_planning",
        "city": None,
    },
    "Event_Guide.pdf": {
        "doc_id": "event_guide",
        "title": "Event Planning Guide",
        "geographic_scope": "International",
        "scope_level": "international",
        "source_priority": 4,
        "category": "event_planning",
        "city": None,
    },
    "Safe_and_legal_event_guidance.pdf": {
        "doc_id": "safe_and_legal_event_guidance",
        "title": "Safe and Legal Event Guidance",
        "geographic_scope": "International",
        "scope_level": "international",
        "source_priority": 4,
        "category": "event_planning",
        "city": None,
    },
    "monash-event-guide.pdf": {
        "doc_id": "monash_event_guide",
        "title": "Monash Event Guide",
        "geographic_scope": "International",
        "scope_level": "international",
        "source_priority": 4,
        "category": "event_planning",
        "city": None,
    },
    # 02_food_catering
    "Guidance_Document_Catering_Sector_19_01_2018(4).pdf": {
        "doc_id": "fssai_guidance_document_catering_sector",
        "title": "FSSAI Guidance Document for Food Safety in Catering Sector",
        "geographic_scope": "India",
        "scope_level": "national",
        "source_priority": 3,
        "category": "catering",
        "city": None,
    },
    # 03_travel_reimbursement
    "Employee Travel Reimbursement Guide _ Financial Affairs.pdf": {
        "doc_id": "employee_travel_reimbursement_guide",
        "title": "Employee Travel Reimbursement Guide",
        "geographic_scope": "Corporate / General",
        "scope_level": "corporate",
        "source_priority": 1,
        "category": "travel_reimbursement",
        "city": None,
    },
    "Small Business International Travel Resource Travel Planner.pdf": {
        "doc_id": "small_business_international_travel_planner",
        "title": "Small Business International Travel Resource Travel Planner",
        "geographic_scope": "International",
        "scope_level": "international",
        "source_priority": 2,
        "category": "travel_reimbursement",
        "city": None,
    },
    # 04_pune_venues
    "Crowne Plaza Pune City Centre - Hotel Meeting Rooms for Rent.pdf": {
        "doc_id": "crowne_plaza_pune_city_centre",
        "title": "Crowne Plaza Pune City Centre - Hotel Meeting Rooms for Rent",
        "geographic_scope": "Pune, Maharashtra, India",
        "scope_level": "city",
        "source_priority": 1,
        "category": "pune_venue",
        "city": "Pune",
    },
    "DES _ Venue Booking.pdf": {
        "doc_id": "des_venue_booking",
        "title": "DES Venue Booking Details",
        "geographic_scope": "Pune, Maharashtra, India",
        "scope_level": "city",
        "source_priority": 1,
        "category": "pune_venue",
        "city": "Pune",
    },
    "DES _ Venue Booking1.pdf": {
        "doc_id": "des_venue_booking_1",
        "title": "DES Venue Booking Terms and Guidelines",
        "geographic_scope": "Pune, Maharashtra, India",
        "scope_level": "city",
        "source_priority": 1,
        "category": "pune_venue",
        "city": "Pune",
    },
    "Event & Meeting Spaces _ Fairfield Pune Kharadi.pdf": {
        "doc_id": "fairfield_pune_kharadi",
        "title": "Event & Meeting Spaces - Fairfield by Marriott Pune Kharadi",
        "geographic_scope": "Pune, Maharashtra, India",
        "scope_level": "city",
        "source_priority": 1,
        "category": "pune_venue",
        "city": "Pune",
    },
    "Event in Pune - DoubleTree by Hilton Pune - Meetings and Events.pdf": {
        "doc_id": "doubletree_hilton_pune",
        "title": "Event in Pune - DoubleTree by Hilton Pune Meetings and Events",
        "geographic_scope": "Pune, Maharashtra, India",
        "scope_level": "city",
        "source_priority": 1,
        "category": "pune_venue",
        "city": "Pune",
    },
    "Hall Booking – DCCIA Pune.pdf": {
        "doc_id": "dccia_pune_hall_booking",
        "title": "Hall Booking - DCCIA Pune",
        "geographic_scope": "Pune, Maharashtra, India",
        "scope_level": "city",
        "source_priority": 1,
        "category": "pune_venue",
        "city": "Pune",
    },
    "Premium Meetings and Conference Halls Pune _ Hyatt Regency Pune.pdf": {
        "doc_id": "hyatt_regency_pune",
        "title": "Premium Meetings and Conference Halls - Hyatt Regency Pune",
        "geographic_scope": "Pune, Maharashtra, India",
        "scope_level": "city",
        "source_priority": 1,
        "category": "pune_venue",
        "city": "Pune",
    },
    # 05_india_general
    "Tourist Guide_CTS2.0_NSQF-3 (2).pdf": {
        "doc_id": "tourist_guide_cts_nsqf",
        "title": "Tourist Guide CTS 2.0 NSQF Level 3",
        "geographic_scope": "India",
        "scope_level": "national",
        "source_priority": 3,
        "category": "india_general",
        "city": None,
    },
    # 06_international_general
    "Business Travel and Work Abroad.pdf": {
        "doc_id": "business_travel_and_work_abroad",
        "title": "Business Travel and Work Abroad Guide",
        "geographic_scope": "International",
        "scope_level": "international",
        "source_priority": 4,
        "category": "international_general",
        "city": None,
    },
}


@dataclass
class DocumentChunk:
    """
    Represents a single chunk of text extracted from a corpus document with provenance metadata.
    """
    chunk_id: str
    doc_id: str
    title: str
    section_id: Optional[str]
    section_title: Optional[str]
    source_file: str
    source_path: str
    source_type: str
    category: str
    page_number: int
    geographic_scope: str
    scope_level: str
    source_priority: int
    text: str
    token_count: int
    char_start: int
    char_end: int
    section: Optional[str] = None
    document_title: Optional[str] = None
    city: Optional[str] = None

    def __post_init__(self):
        if self.section is None:
            self.section = self.section_title
        if self.document_title is None:
            self.document_title = self.title
        if self.city is None and self.category == "pune_venue":
            self.city = "Pune"


class CorpusIndexBuilder:
    """
    Orchestrates recursive PDF discovery, page-by-page text extraction,
    geographic-scope & category metadata tagging, section-aware chunking,
    BM25 indexing, and dense embedding generation.
    """

    def __init__(
        self,
        raw_docs_dir: str = "corpus/raw_docs",
        output_data_dir: str = "data",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        embedding_model_name: str = "text-embedding-004",
        embedding_dimension: int = 768,
        api_key: Optional[str] = None,
    ):
        self.raw_docs_dir = Path(raw_docs_dir)
        self.output_data_dir = Path(output_data_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model_name
        self.embedding_dimension = embedding_dimension
        self.api_key = (
            api_key
            or getattr(settings, "llm_api_key", "")
            or os.getenv("GEMINI_API_KEY", "")
            or os.getenv("GOOGLE_API_KEY", "")
            or os.getenv("LLM_API_KEY", "")
        )
        self.extraction_warnings: List[str] = []

    def _tokenize(self, text: str) -> List[str]:
        """Simple regex tokenizer for BM25 indexing and query matching."""
        return re.findall(r"\w+", text.lower())

    def _derive_doc_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Derives scope, category, and city metadata from registry or heuristics."""
        file_name = file_path.name
        parent_name = file_path.parent.name
        normalized_category = CATEGORY_ALIAS_MAP.get(parent_name, parent_name)

        if file_name in DOCUMENT_REGISTRY:
            meta = dict(DOCUMENT_REGISTRY[file_name])
            if "category" not in meta or meta["category"].startswith("0"):
                meta["category"] = normalized_category
            if "city" not in meta:
                meta["city"] = "Pune" if meta.get("category") == "pune_venue" else None
            return meta

        stem = file_path.stem
        doc_id = re.sub(r"[^\w]+", "_", stem).strip("_").lower()
        title = stem.replace("_", " ").replace("-", " ").title()

        city = "Pune" if normalized_category == "pune_venue" or "pune" in stem.lower() else None
        geo_scope = "Pune, Maharashtra, India" if city == "Pune" else ("India" if "india" in normalized_category else "International")
        scope_level = "city" if city == "Pune" else ("national" if geo_scope == "India" else "international")
        priority = 1 if city == "Pune" else (3 if geo_scope == "India" else 4)

        return {
            "doc_id": doc_id,
            "title": title,
            "geographic_scope": geo_scope,
            "scope_level": scope_level,
            "source_priority": priority,
            "category": normalized_category,
            "city": city,
        }

    def _extract_text_from_pdf(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Extracts text page-by-page from a PDF file using PyMuPDF (with fallback to pypdf).
        Preserves page numbers (1-indexed) and detects section headings where reasonably possible.
        """
        pages: List[Dict[str, Any]] = []

        if pymupdf is not None:
            try:
                doc: Any = pymupdf.open(str(file_path))
                for page_idx, page in enumerate(doc, start=1):
                    text = page.get_text() or ""
                    clean_text = text.strip()
                    if clean_text:
                        lines = [ln.strip() for ln in clean_text.split("\n") if ln.strip()]
                        detected_heading = None
                        if lines:
                            first_line = lines[0]
                            if len(first_line) < 120 and (first_line.isupper() or re.match(r"^(?:Section|Chapter|Part|\d+\.)", first_line, re.IGNORECASE)):
                                detected_heading = first_line

                        pages.append({
                            "page_number": page_idx,
                            "text": clean_text,
                            "detected_heading": detected_heading,
                        })
                return pages
            except Exception as e:
                self.extraction_warnings.append(f"PyMuPDF failed for '{file_path.name}': {e}. Trying pypdf fallback.")

        # Fallback to pypdf
        if pypdf is not None:
            try:
                reader = pypdf.PdfReader(str(file_path))
                for page_idx, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    clean_text = text.strip()
                    if clean_text:
                        lines = [ln.strip() for ln in clean_text.split("\n") if ln.strip()]
                        detected_heading = lines[0] if lines and len(lines[0]) < 120 and (lines[0].isupper() or re.match(r"^(?:Section|Chapter|Part|\d+\.)", lines[0], re.IGNORECASE)) else None
                        pages.append({
                            "page_number": page_idx,
                            "text": clean_text,
                            "detected_heading": detected_heading,
                        })
                return pages
            except Exception as e:
                self.extraction_warnings.append(f"pypdf failed for '{file_path.name}': {e}.")

        return pages

    def load_raw_documents(self) -> List[Dict[str, Any]]:
        """
        Discovers PDF documents in the 6 active folders under `self.raw_docs_dir` and extracts text page-by-page.
        Excludes legacy folders and deprecated documents.
        """
        if not self.raw_docs_dir.exists():
            raise FileNotFoundError(f"Corpus directory not found: {self.raw_docs_dir}")

        # Filter PDFs only from active folders and non-excluded files
        pdf_paths: List[Path] = []
        for p in sorted(self.raw_docs_dir.rglob("*.pdf")):
            if not p.is_file() or p.name.startswith("."):
                continue
            parent_name = p.parent.name
            if parent_name in EXCLUDED_FOLDERS:
                continue
            if parent_name not in CATEGORY_ALIAS_MAP:
                continue
            if p.name in EXCLUDED_FILENAMES or "model-building-bye-laws" in p.name.lower() or "model_building_bye_laws" in p.name.lower():
                continue
            pdf_paths.append(p)

        documents: List[Dict[str, Any]] = []
        seen_doc_ids: Set[str] = set()

        for file_path in pdf_paths:
            meta = self._derive_doc_metadata(file_path)
            doc_id = meta["doc_id"]
            if doc_id in seen_doc_ids:
                doc_id = f"{doc_id}_{meta['category']}"
            seen_doc_ids.add(doc_id)

            extracted_pages = self._extract_text_from_pdf(file_path)
            if not extracted_pages:
                warn_msg = f"PDF '{file_path.relative_to(self.raw_docs_dir)}' has 0 extractable text characters (scanned image or unreadable). Skipped."
                self.extraction_warnings.append(warn_msg)
                print(f"[WARN] {warn_msg}")
                continue

            # Build sections from pages
            sections: List[Dict[str, Any]] = []
            current_heading = meta["title"]

            for p_info in extracted_pages:
                p_num = p_info["page_number"]
                p_text = p_info["text"]
                if p_info["detected_heading"]:
                    current_heading = p_info["detected_heading"]

                sections.append({
                    "section_id": f"p{p_num:03d}",
                    "section_title": current_heading,
                    "page_number": p_num,
                    "text": p_text,
                    "char_start": 0,
                    "char_end": len(p_text),
                })

            full_text = "\n\n".join(p["text"] for p in extracted_pages)

            documents.append({
                "doc_id": doc_id,
                "title": meta["title"],
                "source_file": file_path.name,
                "source_path": str(file_path.relative_to(self.raw_docs_dir)),
                "source_type": "pdf",
                "category": meta["category"],
                "city": meta.get("city"),
                "geographic_scope": meta["geographic_scope"],
                "scope_level": meta["scope_level"],
                "source_priority": meta["source_priority"],
                "pages_count": len(extracted_pages),
                "full_text": full_text,
                "sections": sections,
            })

        return documents

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """
        Splits document pages/sections into granular chunks while strictly preserving
        page-number and document provenance metadata.
        """
        chunks: List[DocumentChunk] = []

        for doc in documents:
            doc_id = doc["doc_id"]
            title = doc["title"]
            source_file = doc["source_file"]
            source_path = doc["source_path"]
            source_type = doc["source_type"]
            category = doc["category"]
            city = doc.get("city")
            geo_scope = doc["geographic_scope"]
            scope_level = doc["scope_level"]
            source_priority = doc["source_priority"]

            for sec in doc["sections"]:
                section_id = sec["section_id"]
                section_title = sec["section_title"]
                page_number = sec["page_number"]
                sec_text = sec["text"]

                # Split section text into sentences without breaking sentences
                sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", sec_text) if s.strip()]
                if not sentences:
                    sentences = [sec_text]

                current_chunk_sentences: List[str] = []
                current_word_count = 0
                chunk_index = 1

                for sentence in sentences:
                    sentence_words = len(sentence.split())

                    if current_word_count + sentence_words > self.chunk_size and current_chunk_sentences:
                        # Flush current chunk
                        chunk_text = " ".join(current_chunk_sentences).strip()
                        if chunk_text:
                            chunk_id = f"{doc_id}_p{page_number:03d}_{chunk_index:02d}"
                            token_count = len(chunk_text.split())
                            char_start = sec_text.find(current_chunk_sentences[0])
                            if char_start < 0:
                                char_start = 0
                            char_end = char_start + len(chunk_text)

                            chunks.append(DocumentChunk(
                                chunk_id=chunk_id,
                                doc_id=doc_id,
                                title=title,
                                section_id=section_id,
                                section_title=section_title,
                                source_file=source_file,
                                source_path=source_path,
                                source_type=source_type,
                                category=category,
                                page_number=page_number,
                                geographic_scope=geo_scope,
                                scope_level=scope_level,
                                source_priority=source_priority,
                                text=chunk_text,
                                token_count=token_count,
                                char_start=char_start,
                                char_end=char_end,
                                section=section_title,
                                document_title=title,
                                city=city,
                            ))
                            chunk_index += 1

                        # Overlap sliding window
                        overlap_sentences: List[str] = []
                        overlap_words = 0
                        for s in reversed(current_chunk_sentences):
                            s_words = len(s.split())
                            if overlap_words + s_words <= self.chunk_overlap:
                                overlap_sentences.insert(0, s)
                                overlap_words += s_words
                            else:
                                break

                        current_chunk_sentences = overlap_sentences + [sentence]
                        current_word_count = sum(len(s.split()) for s in current_chunk_sentences)
                    else:
                        current_chunk_sentences.append(sentence)
                        current_word_count += sentence_words

                if current_chunk_sentences:
                    chunk_text = " ".join(current_chunk_sentences).strip()
                    if chunk_text:
                        chunk_id = f"{doc_id}_p{page_number:03d}_{chunk_index:02d}"
                        token_count = len(chunk_text.split())
                        char_start = sec_text.find(current_chunk_sentences[0])
                        if char_start < 0:
                            char_start = 0
                        char_end = char_start + len(chunk_text)

                        chunks.append(DocumentChunk(
                            chunk_id=chunk_id,
                            doc_id=doc_id,
                            title=title,
                            section_id=section_id,
                            section_title=section_title,
                            source_file=source_file,
                            source_path=source_path,
                            source_type=source_type,
                            category=category,
                            page_number=page_number,
                            geographic_scope=geo_scope,
                            scope_level=scope_level,
                            source_priority=source_priority,
                            text=chunk_text,
                            token_count=token_count,
                            char_start=char_start,
                            char_end=char_end,
                            section=section_title,
                            document_title=title,
                            city=city,
                        ))

        return chunks

    def build_bm25_index(self, chunks: List[DocumentChunk]) -> Any:
        """Tokenizes chunk texts and constructs a BM25 sparse retrieval index with title, scope, city, and category context."""
        if BM25Okapi is None:
            raise ImportError("rank-bm25 is required for BM25 indexing. Install via pip install rank-bm25.")

        tokenized_corpus = [
            self._tokenize(f"{chunk.title} {chunk.category} {chunk.city or ''} {chunk.geographic_scope} {chunk.section_title or ''} {chunk.text}")
            for chunk in chunks
        ]
        bm25 = BM25Okapi(tokenized_corpus)
        return bm25

    def _generate_local_deterministic_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generates deterministic, high-quality normalized dense embeddings locally using
        n-gram TF-IDF projection and feature hashing. Ensures offline portability and
        exact dimensional alignment (N x embedding_dimension) without requiring API keys.
        """
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.preprocessing import normalize

        vectorizer = HashingVectorizer(
            n_features=self.embedding_dimension,
            alternate_sign=True,
            norm=cast(Any, None),
            analyzer="word",
            ngram_range=(1, 2)
        )
        sparse_mat: Any = vectorizer.transform(texts)
        dense_mat = sparse_mat.toarray().astype(np.float32)

        normalized_mat = cast(Any, normalize(dense_mat, norm="l2", axis=1)).astype(np.float32)
        return normalized_mat

    def build_dense_embeddings(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """Generates dense vector embeddings for all chunk texts with title, city, category, and scope context."""
        texts = [
            f"{chunk.title} [{chunk.category}] ({chunk.city or chunk.geographic_scope}) {chunk.section_title or ''}: {chunk.text}"
            for chunk in chunks
        ]
        if not texts:
            return np.empty((0, self.embedding_dimension), dtype=np.float32)

        if self.api_key and self.api_key.strip():
            try:
                from google import genai
                client = genai.Client(api_key=self.api_key)
                all_embeddings: List[List[float]] = []
                batch_size = 50

                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    response = client.models.embed_content(
                        model=self.embedding_model_name,
                        contents=batch_texts
                    )
                    if hasattr(response, "embeddings") and response.embeddings:
                        for emb in response.embeddings:
                            if emb.values is not None:
                                all_embeddings.append(emb.values)
                    else:
                        raise ValueError("Unexpected response format from embedding API.")

                embeddings_arr = np.array(all_embeddings, dtype=np.float32)
                self.used_embedding_backend = "genai"
                return embeddings_arr
            except Exception as e:
                print(f"[WARN] Google GenAI embedding failed ({e}). Falling back to local dense vectorizer.")

        self.used_embedding_backend = "local"
        return self._generate_local_deterministic_embeddings(texts)

    def save_indices(
        self,
        chunks: List[DocumentChunk],
        bm25_index: Any,
        embeddings: np.ndarray,
    ) -> None:
        """Persists chunk metadata, BM25 index, dense embeddings, and manifest."""
        self.output_data_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save metadata JSON
        metadata_path = self.output_data_dir / "chunk_metadata.json"
        metadata_records = [asdict(c) for c in chunks]
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_records, f, indent=2, ensure_ascii=False)

        # 2. Save BM25 index pickle
        bm25_path = self.output_data_dir / "bm25_index.pkl"
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25_index, f)

        # 3. Save dense embeddings numpy matrix
        embeddings_path = self.output_data_dir / "embeddings.npy"
        np.save(embeddings_path, embeddings)

        # 4. Save index manifest
        manifest_path = self.output_data_dir / "index_manifest.json"
        backend = getattr(self, "used_embedding_backend", "local")
        manifest = {
            "embedding_backend": backend,
            "embedding_model": self.embedding_model_name if backend == "genai" else "local-hashing",
            "embedding_dimension": self.embedding_dimension,
            "chunk_count": len(chunks)
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def run(self) -> Dict[str, Any]:
        """Executes the end-to-end corpus indexing pipeline and outputs validation statistics."""
        print(f"=== Starting Active PDF Corpus Indexing Pipeline ===")
        print(f"Source Directory: {self.raw_docs_dir.resolve()}")
        print(f"Target Directory: {self.output_data_dir.resolve()}")

        # 1. Load documents
        documents = self.load_raw_documents()
        total_sections = sum(len(doc["sections"]) for doc in documents)
        total_pages = sum(doc["pages_count"] for doc in documents)
        print(f"Documents loaded: {len(documents)}")
        print(f"Total pages extracted: {total_pages}")
        print(f"Total sections extracted: {total_sections}")

        # 2. Chunk documents
        chunks = self.chunk_documents(documents)
        print(f"Chunks created: {len(chunks)}")

        assert len(chunks) > 0, "No chunks generated from raw documents."
        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk IDs detected."

        for c in chunks:
            assert c.doc_id, f"Chunk {c.chunk_id} missing doc_id"
            assert c.title, f"Chunk {c.chunk_id} missing title"
            assert c.section_title or c.section, f"Chunk {c.chunk_id} missing section"
            assert c.source_file, f"Chunk {c.chunk_id} missing source_file"
            assert c.category, f"Chunk {c.chunk_id} missing category"
            assert c.page_number > 0, f"Chunk {c.chunk_id} has invalid page_number {c.page_number}"
            assert c.geographic_scope, f"Chunk {c.chunk_id} missing geographic_scope"
            assert c.scope_level, f"Chunk {c.chunk_id} missing scope_level"
            assert c.source_priority in {1, 2, 3, 4}, f"Chunk {c.chunk_id} invalid source_priority {c.source_priority}"
            assert c.text.strip(), f"Chunk {c.chunk_id} has empty text"

        # 3. Build BM25 sparse index
        bm25_index = self.build_bm25_index(chunks)
        print(f"BM25 documents: {bm25_index.corpus_size}")
        assert bm25_index.corpus_size == len(chunks), "BM25 document count does not match chunk count."

        # 4. Generate Dense vector embeddings
        embeddings = self.build_dense_embeddings(chunks)
        print(f"Dense index shape: {embeddings.shape} (float32)")
        assert embeddings.shape[0] == len(chunks), "Embedding rows do not match chunk count."
        assert embeddings.shape[1] == self.embedding_dimension, f"Embedding dimension {embeddings.shape[1]} != {self.embedding_dimension}"

        # 5. Persist indices and metadata
        self.save_indices(chunks, bm25_index, embeddings)

        active_folders_list = [
            "01_event_planning",
            "02_food_catering",
            "03_travel_reimbursement",
            "04_pune_venues",
            "05_india_general",
            "06_international_general",
        ]
        excluded_folders_list = [
            "02_crowd_management",
            "03_safety_emergency",
            "04_event_permits_compliance",
            "05_accessibility",
        ]

        alignment_status = (
            "ALIGNED (metadata count == BM25 count == dense count)"
            if len(chunks) == bm25_index.corpus_size == embeddings.shape[0]
            else "MISALIGNED"
        )

        print("\n" + "=" * 65)
        print("CONCISE FINAL CORPUS REBUILD REPORT")
        print("=" * 65)
        print(f"Active Folders:           {', '.join(active_folders_list)}")
        print(f"Excluded Folders:         {', '.join(excluded_folders_list)}")
        print(f"Source Documents:         {len(documents)}")
        print(f"Total Chunks:             {len(chunks)}")
        print(f"Dense Embedding Shape:    {embeddings.shape}")
        print(f"Alignment Status:         {alignment_status}")
        print(f"Extraction Failures:      {len(self.extraction_warnings)}")
        if self.extraction_warnings:
            for w in self.extraction_warnings:
                print(f"  - {w}")
        print("=" * 65 + "\n")

        return {
            "active_folders": active_folders_list,
            "excluded_folders": excluded_folders_list,
            "documents_loaded": len(documents),
            "total_pages": total_pages,
            "sections_extracted": total_sections,
            "chunks_created": len(chunks),
            "embeddings_generated": embeddings.shape[0],
            "dense_index_shape": list(embeddings.shape),
            "bm25_documents": bm25_index.corpus_size,
            "alignment_status": alignment_status,
            "extraction_failures": len(self.extraction_warnings),
            "warnings": self.extraction_warnings,
        }


if __name__ == "__main__":
    builder = CorpusIndexBuilder(
        raw_docs_dir=str(settings.corpus_dir),
        output_data_dir=str(settings.data_dir),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        embedding_model_name=settings.embedding_model,
        embedding_dimension=settings.embedding_dimension,
    )
    builder.run()
