from typing import Any
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_path) -> None:
        self.model_path = model_path
        self.model = SentenceTransformer(model_path, trust_remote_code=True)

    def __call__(self, texts) -> Any:
        embeddings = self.model.encode(texts)
        return embeddings
