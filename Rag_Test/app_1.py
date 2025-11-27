# from __future__ import annotations

# import os
# from abc import ABC, abstractmethod
# from typing import List, Dict, Optional, Set, Any

# import numpy as np
# from pydantic import BaseModel,Field


# # Docling
# from docling.document_converter import DocumentConverter
# from docling_core.types.doc import (
#     DocItemLabel,
#     SectionHeaderItem,
#     TextItem,
#     TableItem,
# )

# # LangChain
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


# # Qdrant
# from qdrant_client import QdrantClient
# from qdrant_client.models import (
#     VectorParams,
#     Distance,
#     PointStruct,
# )

# from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

from src.infrastructure.embeddings.huggingface import HuggingFaceEmbedder
from src.infrastructure.vector_stores.qdrant_db import QdrantImpl
from src.application.services.rag_service import VectorStoreService,run_indexing_service,run_retrieval_service
load_dotenv()

EMBED_MODEL = "models/text-embedding-004"  # modelo típico de Gemini

if __name__ == "__main__":


    #gemini_embedder = GeminiEmbedder()

    hug_embedder = HuggingFaceEmbedder(model_name="all-MiniLM-L6-v2")

    # 2. Crear implementación Qdrant in-memory
    qdrant_impl = QdrantImpl(collection_name="rag_chunks")

    # 3. Crear servicio de Vector Store
    vector_store_service = VectorStoreService(
        embedder=hug_embedder,
        db_impl=qdrant_impl,
    )

   
    # --- FASE 1: INDEXADO ---
    files_paths=os.listdir("data")
    for file_path in files_paths:
        run_indexing_service(
            file_path=f"data/{file_path}",
            vector_store=vector_store_service,
        )
    # --- FASE 2: CONSULTA ---
    query_text = "what is the m/z of  4-Dihydroxyacetophenone"
    results = run_retrieval_service(query_text, vector_store_service, top_k=3)

    for i, chunk in enumerate(results, start=1):
        print(f"\n[Resultado {i}]")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Contenido: {chunk.content}")
        print(f"source_file {chunk.source_file}")
        print(f"page {chunk.page}")
        print(f"type {chunk.type}")


