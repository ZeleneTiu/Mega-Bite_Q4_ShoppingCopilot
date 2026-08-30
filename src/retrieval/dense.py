"""B2 (semantic half): in-memory dense vector index.

Two interchangeable backends behind one interface, so the rest of the
engine never knows which is in use.

``svd``   Latent Semantic Analysis: TF-IDF over the catalog, reduced to a
          few hundred dense dimensions with a truncated SVD. Numpy and
          scikit-learn only, no model download, builds in seconds. This is
          the backend measured in ANNA_dump/log.txt, because the session
          this was developed in could not reach the model hub.

``st``    A sentence-transformer encoder. Better semantics in principle,
          but needs a ~90MB model download on first run. Swap to it with
          ``DenseIndex(catalog, backend="st")`` on any machine that can
          reach huggingface.co.

Both normalise their vectors to unit length, so cosine similarity is a
plain dot product, and both serve pool-restricted queries with a direct
numpy gather. Once the category scope has cut the pool to a couple of
hundred rows, a dot product against those rows beats an index lookup over
50,000; FAISS is kept for the unscoped whole-catalog path only, which is
what the competition rules mean by staying out of heavy vector databases.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .catalog import Catalog

DEFAULT_ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Characters of per-product text handed to the encoder. Long descriptions
# dilute the vector, and the discriminative content sits at the front.
DOC_CHAR_LIMIT = 512
DEFAULT_DIMENSIONS = 256


class DenseIndex:
    """Cosine-similarity search over dense product vectors."""

    def __init__(
        self,
        catalog: Catalog,
        backend: str = "svd",
        dimensions: int = DEFAULT_DIMENSIONS,
        model_name: str = DEFAULT_ST_MODEL,
        cache_dir: str | Path = "ANNA_dump/cache",
        batch_size: int = 256,
        random_state: int = 0,
    ) -> None:
        self.catalog = catalog
        self.backend = backend
        self.dimensions = dimensions
        self.model_name = model_name
        self.batch_size = batch_size
        self.random_state = random_state
        self._model = None        # sentence-transformer, loaded lazily
        self._vectoriser = None   # TF-IDF vectoriser, svd backend
        self._svd = None          # fitted truncated SVD, svd backend
        self.embeddings = self._build(Path(cache_dir))
        self._faiss = self._build_faiss()

    # ------------------------------------------------------------------
    def _document_texts(self) -> list[str]:
        """Compact vector input: identity first, then distinguishing detail."""
        texts: list[str] = []
        for doc_id in range(self.catalog.size):
            parts = [
                self.catalog.fields["title"][doc_id],
                self.catalog.fields["categories"][doc_id],
                self.catalog.fields["store"][doc_id],
                self.catalog.fields["features"][doc_id],
            ]
            texts.append(" | ".join(p for p in parts if p)[:DOC_CHAR_LIMIT])
        return texts

    def _build(self, cache_dir: Path) -> np.ndarray:
        if self.backend == "svd":
            return self._build_svd()
        if self.backend == "st":
            return self._build_sentence_transformer(cache_dir)
        raise ValueError(f"unknown dense backend: {self.backend!r}")

    # --- svd backend ---------------------------------------------------
    def _build_svd(self) -> np.ndarray:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        # min_df=2 drops hapax terms, which are noise in a latent space and
        # roughly halve the vocabulary. Bigrams are kept because product
        # language is full of them ("long sleeve", "stainless steel").
        self._vectoriser = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=2, max_features=200_000,
            sublinear_tf=True, strip_accents="unicode",
        )
        matrix = self._vectoriser.fit_transform(self._document_texts())
        self._svd = TruncatedSVD(
            n_components=self.dimensions, algorithm="randomized",
            n_iter=5, random_state=self.random_state,
        )
        reduced = self._svd.fit_transform(matrix).astype(np.float32)
        # Unit length so cosine similarity becomes a dot product.
        return normalize(reduced).astype(np.float32)

    # --- sentence-transformer backend ----------------------------------
    def _cache_path(self, cache_dir: Path) -> Path:
        digest = hashlib.sha256(
            f"{self.model_name}|{self.catalog.size}|{DOC_CHAR_LIMIT}".encode()
        ).hexdigest()[:16]
        return cache_dir / f"embeddings_{digest}.npy"

    def _build_sentence_transformer(self, cache_dir: Path) -> np.ndarray:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(cache_dir)
        if path.exists():
            return np.load(path)
        model = self._ensure_model()
        vectors = model.encode(
            self._document_texts(), batch_size=self.batch_size,
            convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        np.save(path, vectors)
        return vectors

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    # ------------------------------------------------------------------
    def _build_faiss(self):
        """Flat inner-product index for unscoped queries. Optional."""
        try:
            import faiss
        except ImportError:
            return None
        index = faiss.IndexFlatIP(self.embeddings.shape[1])
        index.add(np.ascontiguousarray(self.embeddings))
        return index

    def encode_query(self, text: str) -> np.ndarray:
        if self.backend == "svd":
            from sklearn.preprocessing import normalize
            reduced = self._svd.transform(self._vectoriser.transform([text]))
            return normalize(reduced).astype(np.float32)[0]
        return self._ensure_model().encode(
            [text], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)[0]

    def score(self, query_text: str, pool: np.ndarray | None = None) -> np.ndarray:
        """Cosine similarity of the query against ``pool`` (or everything)."""
        length = len(pool) if pool is not None else self.catalog.size
        if not query_text.strip():
            return np.zeros(length, dtype=np.float32)
        vector = self.encode_query(query_text)
        if not np.any(vector):
            # Query had no in-vocabulary terms. Abstain rather than return
            # an arbitrary ordering that RRF would then treat as signal.
            return np.zeros(length, dtype=np.float32)
        if pool is None:
            return self.embeddings @ vector
        return self.embeddings[pool] @ vector

    def search_all(self, query_text: str, top_k: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """Whole-catalog nearest neighbours via FAISS, for unscoped browsing."""
        vector = self.encode_query(query_text).reshape(1, -1)
        if self._faiss is not None:
            scores, ids = self._faiss.search(np.ascontiguousarray(vector), top_k)
            return ids[0].astype(np.int32), scores[0].astype(np.float32)
        sims = self.embeddings @ vector[0]
        ids = np.argpartition(-sims, top_k - 1)[:top_k]
        ids = ids[np.argsort(-sims[ids])]
        return ids.astype(np.int32), sims[ids].astype(np.float32)
