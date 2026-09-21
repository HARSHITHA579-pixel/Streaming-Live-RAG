"""
Corpus Index Builder for Streaming RAG

================================================================================
RESPONSIBILITY:
- Read raw documents from `corpus/raw_docs/`.
- Extract text while strictly preserving document IDs, hierarchy, and section information.
- Chunk documents using configured chunk size and overlap strategies.
- Store granular metadata for every chunk (chunk_id, doc_id, section, content, token_count).
- Build the sparse BM25 index and serialize it.
- Generate dense vector embeddings for all chunks via embedding model API/local model.
- Save indices and metadata artifacts into the `data/` directory.

INPUTS:
- Raw text / markdown / PDF documents placed in `corpus/raw_docs/`.
- Indexing hyperparameters from `app.config.Settings` (chunk_size, chunk_overlap, embedding model).

OUTPUTS:
- `data/bm25_index.pkl`: Serialized BM25 index with tokenized corpus vocabulary.
- `data/embeddings.npy`: Dense embedding vectors (N x D float32 matrix).
- `data/chunk_metadata.json`: Chunk-to-document mapping containing doc_id, section title,
  chunk_id, text snippet, and token offsets.

CONNECTED COMPONENTS:
- `app.config`: Reads indexing parameters (chunk sizes, embedding models, output paths).
- `app.retriever`: Consumes the generated BM25 index, embedding matrix, and chunk metadata
  during runtime hybrid search.

WHY THIS ARCHITECTURE:
- The corpus is the sole ground truth of factual evidence.
- Preserving section metadata during indexing enables high-precision citations
  (e.g., "Doc_12 §2") required by the downstream claim verifier and synthesis engine.
================================================================================
"""

import os
import re
import json
import pickle
import hashlib
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, cast
from pathlib import Path

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    from app.config import settings, Settings
except ImportError:
    # Fallback if imported outside package context
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
    text: str
    token_count: int
    char_start: int
    char_end: int
    section: Optional[str] = None

    def __post_init__(self):
        if self.section is None:
            self.section = self.section_title


class CorpusIndexBuilder:
    """
    Orchestrates document parsing, chunking, metadata tracking, BM25 indexing,
    and dense embedding generation.
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
        self.api_key = api_key or getattr(settings, "llm_api_key", "") or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "") or os.getenv("LLM_API_KEY", "")

    def _tokenize(self, text: str) -> List[str]:
        """Simple regex tokenizer for BM25 indexing and query matching."""
        return re.findall(r"\w+", text.lower())

    def load_raw_documents(self) -> List[Dict[str, Any]]:
        """
        Reads and extracts raw text and section structures from files in `self.raw_docs_dir`.
        Supports .md and .txt files without requiring an external LLM parser.
        """
        if not self.raw_docs_dir.exists():
            raise FileNotFoundError(f"Corpus directory not found: {self.raw_docs_dir}")

        documents: List[Dict[str, Any]] = []
        supported_extensions = {".md", ".txt"}

        # Collect and sort all supported raw document files
        file_paths = sorted([
            p for p in self.raw_docs_dir.iterdir()
            if p.is_file() and p.suffix.lower() in supported_extensions and not p.name.startswith(".")
        ])

        for file_path in file_paths:
            content = file_path.read_text(encoding="utf-8").strip()
            if not content:
                continue

            doc_id = file_path.stem
            source_file = file_path.name

            # Extract Document Title from first '# ' heading or fallback to formatted filename
            title = doc_id.replace("_", " ").title()
            title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
            if title_match:
                title = title_match.group(1).strip()

            # Parse sections delineated by '## ' headings
            section_pattern = re.compile(r"^##\s+(.+)$", re.MULTILINE)
            section_matches = list(section_pattern.finditer(content))

            sections: List[Dict[str, Any]] = []

            if not section_matches:
                # No '## ' sub-sections found; treat entire document as one section
                body_text = content
                if title_match:
                    body_text = content[title_match.end():].strip()
                sections.append({
                    "section_id": "sec_01",
                    "section_title": title,
                    "text": body_text,
                    "char_start": 0,
                    "char_end": len(content)
                })
            else:
                # Check for any introductory text before first '## '
                first_match = section_matches[0]
                intro_text = content[:first_match.start()].strip()
                if title_match:
                    intro_text = content[title_match.end():first_match.start()].strip()
                if intro_text:
                    sections.append({
                        "section_id": "sec_00",
                        "section_title": "Overview",
                        "text": intro_text,
                        "char_start": 0,
                        "char_end": first_match.start()
                    })

                for idx, match in enumerate(section_matches, start=1):
                    sec_title = match.group(1).strip()
                    sec_start = match.end()
                    sec_end = section_matches[idx].start() if idx < len(section_matches) else len(content)
                    sec_text = content[sec_start:sec_end].strip()

                    if sec_text:
                        sections.append({
                            "section_id": f"sec_{idx:02d}",
                            "section_title": sec_title,
                            "text": sec_text,
                            "char_start": match.start(),
                            "char_end": sec_end
                        })

            documents.append({
                "doc_id": doc_id,
                "title": title,
                "source_file": source_file,
                "full_text": content,
                "sections": sections
            })

        return documents

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """
        Splits documents into granular, section-aware chunks while preserving sentence boundaries
        and provenance metadata.
        """
        chunks: List[DocumentChunk] = []

        for doc in documents:
            doc_id = doc["doc_id"]
            title = doc["title"]
            source_file = doc["source_file"]
            chunk_counter = 1

            for sec in doc["sections"]:
                section_id = sec["section_id"]
                section_title = sec["section_title"]
                sec_text = sec["text"]
                sec_char_start = sec["char_start"]

                # Split section text into sentences / paragraphs without breaking mid-sentence
                sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", sec_text) if s.strip()]
                if not sentences:
                    sentences = [sec_text]

                current_chunk_sentences: List[str] = []
                current_word_count = 0

                for sentence in sentences:
                    sentence_words = len(sentence.split())

                    if current_word_count + sentence_words > self.chunk_size and current_chunk_sentences:
                        # Flush current chunk
                        chunk_text = " ".join(current_chunk_sentences).strip()
                        if chunk_text:
                            chunk_id = f"{doc_id}_{chunk_counter:02d}"
                            token_count = len(chunk_text.split())
                            char_start_in_sec = sec_text.find(current_chunk_sentences[0])
                            char_start = sec_char_start + (char_start_in_sec if char_start_in_sec >= 0 else 0)
                            char_end = char_start + len(chunk_text)

                            chunks.append(DocumentChunk(
                                chunk_id=chunk_id,
                                doc_id=doc_id,
                                title=title,
                                section_id=section_id,
                                section_title=section_title,
                                source_file=source_file,
                                text=chunk_text,
                                token_count=token_count,
                                char_start=char_start,
                                char_end=char_end,
                                section=section_title
                            ))
                            chunk_counter += 1

                        # Compute overlap sentences for sliding window
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
                        chunk_id = f"{doc_id}_{chunk_counter:02d}"
                        token_count = len(chunk_text.split())
                        char_start_in_sec = sec_text.find(current_chunk_sentences[0])
                        char_start = sec_char_start + (char_start_in_sec if char_start_in_sec >= 0 else 0)
                        char_end = char_start + len(chunk_text)

                        chunks.append(DocumentChunk(
                            chunk_id=chunk_id,
                            doc_id=doc_id,
                            title=title,
                            section_id=section_id,
                            section_title=section_title,
                            source_file=source_file,
                            text=chunk_text,
                            token_count=token_count,
                            char_start=char_start,
                            char_end=char_end,
                            section=section_title
                        ))
                        chunk_counter += 1

        return chunks

    def build_bm25_index(self, chunks: List[DocumentChunk]) -> Any:
        """
        Tokenizes chunk texts and constructs a BM25 sparse retrieval index.
        """
        if BM25Okapi is None:
            raise ImportError("rank-bm25 is required for BM25 indexing. Install via pip install rank-bm25.")

        tokenized_corpus = [self._tokenize(chunk.text) for chunk in chunks]
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

        # Hash words and character n-grams into the target embedding dimension
        vectorizer = HashingVectorizer(
            n_features=self.embedding_dimension,
            alternate_sign=True,
            norm=cast(Any, None),
            analyzer="word",
            ngram_range=(1, 2)
        )
        sparse_mat: Any = vectorizer.transform(texts)
        dense_mat = sparse_mat.toarray().astype(np.float32)

        # L2-normalize vectors for cosine similarity
        normalized_mat = normalize(dense_mat, norm="l2", axis=1).astype(np.float32)
        return normalized_mat

    def build_dense_embeddings(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """
        Generates dense vector embeddings for all chunk texts.
        Uses Google GenAI API (text-embedding-004) when an API key is configured,
        with seamless local fallback if running offline or in mock environments.
        """
        texts = [chunk.text for chunk in chunks]
        if not texts:
            return np.empty((0, self.embedding_dimension), dtype=np.float32)

        # Attempt to use Google GenAI embedding if credentials are provided
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

        # Local deterministic fallback
        self.used_embedding_backend = "local"
        return self._generate_local_deterministic_embeddings(texts)

    def save_indices(
        self,
        chunks: List[DocumentChunk],
        bm25_index: Any,
        embeddings: np.ndarray,
    ) -> None:
        """
        Persists chunk metadata, BM25 index, and dense embeddings to the `data/` directory.
        """
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
        """
        Executes the end-to-end corpus indexing pipeline and outputs validation statistics.
        """
        print(f"=== Starting Corpus Indexing Pipeline ===")
        print(f"Source Directory: {self.raw_docs_dir.resolve()}")
        print(f"Target Directory: {self.output_data_dir.resolve()}")

        # 1. Load documents
        documents = self.load_raw_documents()
        total_sections = sum(len(doc["sections"]) for doc in documents)
        print(f"Documents loaded: {len(documents)}")
        print(f"Sections extracted: {total_sections}")

        # 2. Chunk documents
        chunks = self.chunk_documents(documents)
        print(f"Chunks created: {len(chunks)}")

        # Validation assertions
        assert len(chunks) > 0, "No chunks generated from raw documents."
        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk IDs detected."
        for c in chunks:
            assert c.doc_id, f"Chunk {c.chunk_id} missing doc_id"
            assert c.title, f"Chunk {c.chunk_id} missing title"
            assert c.section_title or c.section, f"Chunk {c.chunk_id} missing section"
            assert c.source_file, f"Chunk {c.chunk_id} missing source_file"
            assert c.text.strip(), f"Chunk {c.chunk_id} has empty text"

        # 3. Build BM25 sparse index
        bm25_index = self.build_bm25_index(chunks)
        print(f"BM25 documents: {bm25_index.corpus_size}")
        assert bm25_index.corpus_size == len(chunks), "BM25 document count does not match chunk count."

        # 4. Generate Dense vector embeddings
        embeddings = self.build_dense_embeddings(chunks)
        print(f"Embeddings generated: {embeddings.shape[0]}")
        print(f"Dense index size: {embeddings.shape} (float32)")
        assert embeddings.shape[0] == len(chunks), "Embedding rows do not match chunk count."
        assert embeddings.shape[1] == self.embedding_dimension, f"Embedding dimension {embeddings.shape[1]} != {self.embedding_dimension}"

        # 5. Persist indices and metadata
        self.save_indices(chunks, bm25_index, embeddings)
        print(f"Index build completed successfully.")
        print(f"==========================================")

        return {
            "documents_loaded": len(documents),
            "sections_extracted": total_sections,
            "chunks_created": len(chunks),
            "embeddings_generated": embeddings.shape[0],
            "dense_index_shape": list(embeddings.shape),
            "bm25_documents": bm25_index.corpus_size,
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

