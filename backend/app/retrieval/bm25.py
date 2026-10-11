import re
import math
from typing import List, Tuple, Dict
from app.api.schemas import ResearchPaper, Patent
from app.logging_config import logger

def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercased alphanumeric words."""
    if not text:
        return []
    return re.findall(r'\b\w+\b', text.lower())

class PurePythonBM25:
    """Pure Python BM25 implementation fallback."""
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 1.0
        self.doc_freqs = []
        self.idf = {}
        self._initialize(corpus)

    def _initialize(self, corpus: List[List[str]]):
        nd = {}
        for doc in corpus:
            frequencies = {}
            for word in doc:
                frequencies[word] = frequencies.get(word, 0) + 1
            self.doc_freqs.append(frequencies)
            for word in frequencies:
                nd[word] = nd.get(word, 0) + 1

        for word, freq in nd.items():
            idf = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)
            self.idf[word] = max(idf, 0.0)

    def get_scores(self, query: List[str]) -> List[float]:
        scores = [0.0] * self.corpus_size
        for word in query:
            if word not in self.idf:
                continue
            word_idf = self.idf[word]
            for idx, doc_freq in enumerate(self.doc_freqs):
                freq = doc_freq.get(word, 0)
                if freq == 0:
                    continue
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * (self.doc_len[idx] / self.avgdl))
                scores[idx] += word_idf * (numerator / denominator)
        return scores

class BM25Retriever:
    """Keyword-based BM25 retrieval baseline."""
    def __init__(self):
        self.doc_ids: List[str] = []
        self.doc_types: List[str] = []  # "paper" or "patent"
        self.corpus_tokens: List[List[str]] = []
        self.bm25 = None

    def index_documents(self, papers: List[ResearchPaper], patents: List[Patent]):
        self.doc_ids = []
        self.doc_types = []
        self.corpus_tokens = []

        for paper in papers:
            text = f"{paper.title} {paper.abstract or ''} {' '.join(paper.keywords)}"
            tokens = tokenize(text)
            self.doc_ids.append(paper.paper_id)
            self.doc_types.append("paper")
            self.corpus_tokens.append(tokens)

        for patent in patents:
            claims_text = " ".join(patent.claims or [])
            text = f"{patent.title} {patent.abstract or ''} {claims_text} {' '.join(patent.classifications)}"
            tokens = tokenize(text)
            self.doc_ids.append(patent.patent_id)
            self.doc_types.append("patent")
            self.corpus_tokens.append(tokens)

        if self.corpus_tokens:
            try:
                from rank_bm25 import BM25Plus
                self.bm25 = BM25Plus(self.corpus_tokens)
            except Exception:
                self.bm25 = PurePythonBM25(self.corpus_tokens)
            logger.info(f"Indexed {len(self.doc_ids)} total documents in BM25 index.")


    def search(self, query: str, doc_type_filter: str = None, top_k: int = 10) -> List[Tuple[str, float, str]]:
        """Search BM25 index. Returns list of (doc_id, score, doc_type)."""
        if not self.bm25 or not self.corpus_tokens:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        results = []

        for idx, score in enumerate(scores):
            doc_id = self.doc_ids[idx]
            doc_type = self.doc_types[idx]

            if doc_type_filter and doc_type != doc_type_filter:
                continue

            if score > 0:
                results.append((doc_id, float(score), doc_type))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
