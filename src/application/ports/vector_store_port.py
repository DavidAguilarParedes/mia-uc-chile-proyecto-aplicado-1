import numpy as np 
from abc import ABC, abstractmethod
from typing import List,Dict
from src.domain.models import ProcessedChunk
from pydantic import BaseModel, Field



class VectorStoreImpl(ABC):
    @abstractmethod
    def index_data(self, vectors: np.ndarray, metadatas: List[Dict]) -> None:
        pass

    @abstractmethod
    def query_data(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        pass

class RetrievalStrategy(ABC):
    @abstractmethod
    def retrieve_context(self, query: str, filters: Dict, top_k: int
                         ) -> List[ProcessedChunk]:
        pass

class HybridSearchStrategy(RetrievalStrategy):
    def __init__(self, vector_store: VectorStoreImpl):
        self._vector_store = vector_store

    def retrieve_context(self, query: str, filters: Dict, top_k: int = 5
                         ) -> List[ProcessedChunk]:
        # Lógica para combinar búsqueda vectorial y textual
        print(f"Realizando búsqueda híbrida (componente base).")
        results_dict = self._vector_store.query_data(query, filters)
        return [ProcessedChunk(**r) for r in results_dict][:top_k]
    