
import numpy as np
from typing import List
from src.application.ports.vector_store_port import VectorStoreImpl

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
)
class QdrantImpl(VectorStoreImpl):

    def __init__(self, collection_name: str = "rag_chunks"):
        self.client = QdrantClient(":memory:")
        self.collection_name = collection_name
        self._collection_created = False
        self._next_id = 1

    def _ensure_collection(self, vector_size: int):
        if self._collection_created:
            return

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        self._collection_created = True
        print(f"[Qdrant] Colección '{self.collection_name}' creada con dim={vector_size}")

    def index_data(self, vectors: np.ndarray, metadata: list[dict]) -> None:
        if len(vectors) == 0:
            return

        self._ensure_collection(vector_size=vectors.shape[1])

        points = []
        for vec, meta in zip(vectors, metadata):
            points.append(
                PointStruct(
                    id=self._next_id,
                    vector=vec.tolist(),
                    payload=meta,
                )
            )
            self._next_id += 1

        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"[Qdrant] Indexados {len(points)} puntos.")

    def query_data(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Usa el Query API moderno: client.query_points(...).
        """
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),  # <--- el vector de consulta
            limit=top_k,
            with_payload=True,
        )

        out: list[dict] = []
        # result.points es la lista de ScoredPoint
        for p in result.points:
            payload = dict(p.payload or {})
            payload["score"] = p.score
            out.append(payload)

        print(f"[Qdrant] Recuperados {len(out)} resultados.")
        return out
