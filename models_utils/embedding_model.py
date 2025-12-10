from typing import Any
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_path, batch_size=4) -> None:
        self.model_path = model_path
        self.model = SentenceTransformer(model_path, trust_remote_code=True)
        self.batch_size = batch_size

    def __call__(self, texts) -> Any:
        embeddings = self.model.encode(texts)
        return embeddings
