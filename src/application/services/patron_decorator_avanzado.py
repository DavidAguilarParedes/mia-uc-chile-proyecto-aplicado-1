from typing import List,Dict
from src.domain.models import ProcessedChunk
from abc import ABC, abstractmethod

#base del decorator 
class RetrievalDecorator(RetrievalStrategy):
    def __init__(self, wrapped_strategy: RetrievalStrategy):
        self._wrapped_strategy = wrapped_strategy

    def retrieve_context(self, query: str, filters: Dict, top_k: int = 5
                         ) -> List[ProcessedChunk]:
        return self._wrapped_strategy.retrieve_context(query, filters, top_k=top_k)

# Decorator A: Reranking )post-retrieval)
class RerankingDecorator(RetrievalDecorator):
    def __init__(
        self,
        wrapped_strategy: RetrievalStrategy,
        reranker: RerankerService,
    ):
        super().__init__(wrapped_strategy)
        self.reranker_service = reranker_service

    def retrieve_context(self, query: str, filters: Dict, top_k: int = 5
                         ) -> List[ProcessedChunk]:
        # Paso 1: Obtener chunks usando la estrategia envuelta
        chunks_candidatods: List[ProcessedChunk] = self._wrapped_strategy.retrieve_context(
            query, filters, top_k=top_k * 3
        )

        #2. se agrega reranking
        texts_to_rank = [c.content for c in chunks_candidatods]
        scores = self.reranker_service.rerank(query, texts_to_rank)

        #3. Asocia scores a los objetos pydantic (enriquecimiento pydantic)
        scored_chunks = []
        for chunk, score in zip[tuple[ProcessedChunk, float]](chunks_candidatods, scores):
            chunk.rerank_score = score  # Agregar atributo dinámicamente
            scored_chunks.append(chunk)

        #reordena del mayor a menor
        scored_chunks.sort(key=lambda c: c.rerank_score, reverse=True)
        print(
            f" Reordenados. Top scores: {scored_chunks[0].rerank_score:.2f}"
        )

        return scored_chunks[:top_k]
    

    # Decorator B: Context Repacker 
    class ContextRepackerDecorator(RetrievalDecorator):
        def retrieve_context(
                self, query: str, filters: Dict, top_k: int
                ) -> List[ProcessedChunk]:
            # Paso 1: Obtiene chuns ordenados ( de reranker)
            chunks: List[ProcessedChunk] = self._wrapped.retrieve_context(
                query, filters, top_k=top_k
            )

            if len(chunks) < 3:
                return chunks  
            
            print( f" Aplicando Sides Repacking para {len(chunks)} chunks.")

            # Paso 2: Reempaquetar contexto
            top_chunk = chunks[0]
            rest = chunks[1:]

            #nueva estructura para el prompt
            repacked_chunks = [top_chunk] + rest

            return repacked_chunks

