from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class SearchResult:
    item: dict
    score: float


class TfidfRetriever:
    """Small local retriever that works well enough for a dependency-light MVP."""

    def __init__(self, items: Iterable[dict], text_field: str = "text_description") -> None:
        self.items = list(items)
        self.text_field = text_field
        self.documents = [self._document_text(item) for item in self.items]
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            lowercase=False,
        )
        self.matrix = self.vectorizer.fit_transform(self.documents)

    def _document_text(self, item: dict) -> str:
        parts: list[str] = []
        for key in ("name", "syndrome_id", "formula_id", "herb_name", self.text_field):
            value = item.get(key)
            if isinstance(value, str):
                parts.append(value)
        for key in ("core_symptoms", "secondary_symptoms", "targets_op_related", "pathways"):
            value = item.get(key)
            if isinstance(value, list):
                parts.extend(str(part) for part in value)
        return " ".join(parts)

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        if not query.strip():
            return []

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        ranked_indexes = scores.argsort()[::-1][:top_k]

        return [
            SearchResult(item=self.items[index], score=round(float(scores[index]), 4))
            for index in ranked_indexes
        ]


def filter_formulas_by_syndrome(formulas: list[dict], syndrome_id: str) -> list[dict]:
    return [
        formula
        for formula in formulas
        if _matches_syndrome(formula.get("indication_syndrome"), syndrome_id)
    ]


def _matches_syndrome(indication_syndrome: object, syndrome_id: str) -> bool:
    if isinstance(indication_syndrome, str):
        return indication_syndrome == syndrome_id
    if isinstance(indication_syndrome, list):
        return syndrome_id in indication_syndrome
    return False
