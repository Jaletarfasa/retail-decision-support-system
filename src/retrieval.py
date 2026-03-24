from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import numpy as np
import pandas as pd


class LocalTfidfRetriever:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_names: list[str] = []
        self.doc_texts: list[str] = []
        self.doc_matrix = None

    def build_index(self, doc_dir: str | Path) -> None:
        doc_dir = Path(doc_dir)
        paths = sorted(doc_dir.glob("*.txt"))
        self.doc_names = [p.name for p in paths]
        self.doc_texts = [p.read_text(encoding="utf-8") for p in paths]
        if self.doc_texts:
            self.doc_matrix = self.vectorizer.fit_transform(self.doc_texts)

    def search(self, query: str, top_k: int = 3) -> pd.DataFrame:
        if self.doc_matrix is None:
            return pd.DataFrame(columns=["doc_name", "score", "text"])
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.doc_matrix).flatten()
        idx = np.argsort(-sims)[:top_k]
        return pd.DataFrame({"doc_name": [self.doc_names[i] for i in idx], "score": [float(sims[i]) for i in idx], "text": [self.doc_texts[i] for i in idx]})
