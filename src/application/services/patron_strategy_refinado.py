from src.domain.models import ProcessedChunk
from abc import ABC, abstractmethod
from pydantic import BaseModel,Field
from typing import List, Dict, Optional, Set, Any

class QueryProcessingStrategy(ABC):
    @abstractmethod
    def process_query(self, query: str) -> str:
        pass

class QueryRewritingStrategy(QueryProcessingStrategy):
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def process_query(self, query: str) -> FilterSuggestion:
        """" Retorna un objeto pydantic con la query optimizada y filtros"""
        return self.llm_service.generate_structured(
            f"Optimizar esta consulta científica: {query}"
            )
    
class QueryOptimizerRetriever(RetrievalStrategy):
    """" Strategy refinado + Contexto: Une el output pydantic del LLM con
     el input del retrievas strategy (Ragas) """
    
    def __init__(
        self,
        query_processor: QueryProcessingStrategy,
        retrieval_strategy: RetrievalStrategy,
    ):
        self.pocessor= query_processor
        self.retriever = retrieval_strategy

    def retrieve_context(self, query: str, filters: Dict, top_k: int = 5
                         ) -> List[ProcessedChunk]:
        structured_query_output = self.pocessor.process_query(query)

        combined_filters = {**filters, **structured_query_output.metadata_filters}

        print(
            f"Query reescrita: {structured_query_output.rewritten_query}'. Filtros: {combined_filters}"
        )

        return self.retriever.retrieve_context(
            structured_query_output.rewritten_query,
            combined_filters,
            top_k=top_k
        )