from typing import Dict, List

import torch
from langchain_milvus.utils.sparse import BaseSparseEmbedding


def _sparse_to_dict(sparse_row) -> Dict[int, float]:
    """Convert a sparse row to a dict, handling both csr_array (.indices) and coo_array (.col)."""
    if hasattr(sparse_row, 'indices'):
        indices = sparse_row.indices.tolist()
    elif hasattr(sparse_row, 'col'):
        indices = sparse_row.col.tolist()
    else:
        raise AttributeError(f"Unsupported sparse format: {type(sparse_row)}")
    return {k: v for k, v in zip(indices, sparse_row.data.tolist())}


class SpladeSparseEmbedding(BaseSparseEmbedding):
    """Sparse embedding model based on Splade.

    This class uses the Splade embedding model in Milvus model to implement sparse vector embedding.
    This model requires pymilvus[model] to be installed.
    `pip install pymilvus[model]`
    For more information please refer to:
    https://milvus.io/docs/embed-with-splade.md
    """

    def __init__(self, model_name: str = "naver/splade-cocondenser-selfdistil"):
        from pymilvus.model.sparse import SpladeEmbeddingFunction  # type: ignore

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.splade = SpladeEmbeddingFunction(model_name=model_name, device=self.device)

    def embed_query(self, query: str) -> Dict[int, float]:
        res = self.splade.encode_queries([query])
        return _sparse_to_dict(res[0])

    def embed_documents(self, texts: List[str]) -> List[Dict[int, float]]:
        res = self.splade.encode_documents(texts)
        return [_sparse_to_dict(row) for row in res]

