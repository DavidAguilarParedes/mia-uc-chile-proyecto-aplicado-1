# src/application/services/dataset_generation_service.py
import os
from pathlib import Path
from src.application.ports.loader_port import AbstractLoader
from src.infrastructure.evaluation.ragas_generator import RagasLocalGenerator
from src.infrastructure.llm.local_llm_factory import LocalResourcesFactory
from langchain_core.documents import Document

class DatasetGenerationService:
    def __init__(self, loader: AbstractLoader):
        self.loader = loader
        
        # Inicializamos recursos locales (Ollama + HF)
        self.llm = LocalResourcesFactory.get_generator_llm(model_name="qwen2.5:1.5b")
        self.embedder = LocalResourcesFactory.get_embeddings()
        self.generator = RagasLocalGenerator(self.llm, self.embedder)

    def run(self, input_file: str, output_dir: str = "evals/datasets", test_size: int = 5):
        # 1. Cargar Documentos usando tu Loader existente
        # Tu loader devuelve List[ProcessedChunk], necesitamos convertirlo a LangChain Document para Ragas
        print(f"[Service] Cargando: {input_file}")
        chunks = self.loader.load_and_chunk(input_file)
        
        # Adaptador: ProcessedChunk -> LangChain Document
        langchain_docs = [
            Document(page_content=c.content, metadata=c.metadata) 
            for c in chunks
        ]

        # 2. Generar Dataset
        df = self.generator.generate_testset(langchain_docs, test_size=test_size)

        # 3. Guardar
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        base_name = Path(input_file).stem
        output_path = os.path.join(output_dir, f"{base_name}_golden_dataset.csv")
        
        df.to_csv(output_path, index=False)
        print(f"✅ Dataset guardado en: {output_path}")
        return df