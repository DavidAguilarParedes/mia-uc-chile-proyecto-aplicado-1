# src/infrastructure/evaluation/ragas_generator.py
import pandas as pd
from typing import List
from langchain_core.documents import Document as LangChainDocument

# Ragas Imports
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import (
    apply_transforms, 
    HeadlinesExtractor, 
    HeadlineSplitter, 
    KeyphrasesExtractor
)
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
from ragas.testset import TestsetGenerator

class RagasLocalGenerator:
    def __init__(self, generator_llm, generator_embeddings):
        self.llm = generator_llm
        self.embeddings = generator_embeddings
        
    def generate_testset(self, docs: List[LangChainDocument], test_size: int = 10) -> pd.DataFrame:
        print("\n⚙️ [Ragas] Creando Knowledge Graph base...")
        
        # 1. Crear Grafo
        kg = KnowledgeGraph()
        for doc in docs:
            kg.nodes.append(
                Node(
                    type=NodeType.DOCUMENT,
                    properties={
                        "page_content": doc.page_content, 
                        "document_metadata": doc.metadata
                    }
                )
            )
            
        # 2. Configurar Transformaciones (Usando el LLM Local)
        print("\n🛠️ [Ragas] Aplicando Transforms (esto puede tardar con CPU/LLM Local)...")
        transforms = [
            HeadlinesExtractor(llm=self.llm, max_num=5), # Reduje max_num para velocidad local
            HeadlineSplitter(max_tokens=500), # Tokens reducidos para manejo de memoria
            KeyphrasesExtractor(llm=self.llm)
        ]
        
        apply_transforms(kg, transforms=transforms)
        
        # 3. Definir Personas
        personas = [
            Persona(
                name="Analista Junior",
                role_description="Analista principiante que necesita identificar metabolitos básicos y entender datos m/z.",
            ),
            Persona(
                name="Químico Experto",
                role_description="Experto interesado en isómeros, estructuras complejas y rutas biosintéticas.",
            )
        ]

        # 4. Configurar Synthesizers
        query_distribution = [
            (SingleHopSpecificQuerySynthesizer(llm=self.llm, property_name="headlines"), 0.5),
            (SingleHopSpecificQuerySynthesizer(llm=self.llm, property_name="keyphrases"), 0.5),
        ]

        # 5. Generar
        print(f"\n🚀 [Ragas] Generando {test_size} preguntas sintéticas...")
        generator = TestsetGenerator(
            llm=self.llm,
            embedding_model=self.embeddings,
            knowledge_graph=kg,
            persona_list=personas,
        )
        
        testset = generator.generate(testset_size=test_size, query_distribution=query_distribution)
        return testset.to_pandas()