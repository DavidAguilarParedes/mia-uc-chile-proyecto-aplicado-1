import os
import time
import pandas as pd
import asyncio
import streamlit as st
from dotenv import load_dotenv

from src.infrastructure.embeddings.huggingface import HuggingFaceEmbedder
from src.infrastructure.vector_stores.qdrant_db import QdrantImpl
from src.application.services.rag_service import VectorStoreService, run_indexing_service, run_retrieval_service
from src.infrastructure.llm.local_llm_factory import LocalResourcesFactory

from datasets import Dataset 
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness

# Cargar variables de entorno
load_dotenv()

# ==============================================================================

QDRANT_PATH = "qdrant_storage" # La misma ruta  que run_eval.py
DATASET_PATH = "datasets/golden_dataset.csv"

st.set_page_config(page_title="Hito 1: Clean RAG Architecture", layout="wide")

# ==============================================================================
# 1. CAPA DE APLICACIÓN (Inicialización)
# ==============================================================================


@st.cache_resource
def get_vector_service() -> VectorStoreService:
    """
    Instancia los adaptadores y el servicio de aplicación.
    Usa persistencia en disco para compartir datos con los scripts.
    """
    # 1. Adaptador de Embeddings
    embedder = HuggingFaceEmbedder(model_name="all-MiniLM-L6-v2")
    
    # 2. Adaptador de Base de Datos Vectorial (CON PERSISTENCIA)
    db_impl = QdrantImpl(collection_name="rag_chunks", path=QDRANT_PATH)
    
    # 3. Inyección de dependencias
    service = VectorStoreService(embedder=embedder, db_impl=db_impl)
    return service

# ==============================================================================
# 2. SISTEMA DE BENCHMARK REAL
# ==============================================================================

class BaselineEvaluator:
    """
    Evaluador real usando Ragas y Local LLM.
    """
    def __init__(self, rag_service: VectorStoreService):
        self.rag_service = rag_service

    def run_benchmark(self):
        # 1. Verificar Dataset
        if not os.path.exists(DATASET_PATH):
            return None, "❌ No se encontró 'datasets/golden_dataset.csv'. Ejecuta primero 'scripts/generate_dataset.py'."

        # 2. Cargar Dataset
        df = pd.read_csv(DATASET_PATH)
        
        # Validar columnas necesarias
        if 'user_input' not in df.columns or 'reference' not in df.columns:
            return None, "❌ El CSV debe tener columnas 'user_input' y 'reference'."

        questions = df['user_input'].tolist()
        ground_truths = df['reference'].tolist()
        
        answers = []
        contexts = []

        # 3. Correr Inferencia (Retrieval)
        progress_bar = st.progress(0)
        status = st.empty()
        
        total = len(questions)
        for i, q in enumerate(questions):
            status.text(f"Evaluando consulta {i+1}/{total}: {q[:50]}...")
            
            # Llamada al servicio RAG
            results = run_retrieval_service(q, self.rag_service, top_k=3)
            
            # Preparar contexto para Ragas
            retrieved_text = [c.content for c in results]
            
            # Simular respuesta (tomando el top 1 o un mensaje genérico)
            generated_answer = results[0].content if results else "No information found."
            
            answers.append(generated_answer)
            contexts.append(retrieved_text)
            progress_bar.progress((i + 1) / total)

        # 4. Preparar Dataset Ragas
        eval_data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        eval_dataset = Dataset.from_dict(eval_data)

        # 5. Configurar LLM Local para Evaluación
        status.text("Calculando métricas con Ragas (esto puede tardar)...")
        
        llm = LocalResourcesFactory.get_generator_llm("qwen2.5:1.5b")
        embeddings = LocalResourcesFactory.get_embeddings()

        # 6. Ejecutar Ragas
        # Nota: evaluate es asíncrono internamente a veces, pero la API principal bloquea.
        result = evaluate(
            eval_dataset,
            metrics=[answer_relevancy, faithfulness],
            llm=llm, 
            embeddings=embeddings
        )

        status.empty()
        progress_bar.empty()
        
        # Retornar DataFrame de resultados y métrica global
        return result.to_pandas(), result

# ==============================================================================
# 3. INTERFAZ GRÁFICA
# ==============================================================================

def main():
    st.title("🧪 Hito 1: Clean Architecture RAG")
    st.markdown("Implementación Hexagonal con Evaluación Local (Ollama + Ragas).")

    # Obtener el servicio (Singleton)
    rag_service = get_vector_service()

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Gestión de Datos")
        
        data_folder = "data"
        st.info(f"Usando almacenamiento en: `{QDRANT_PATH}/`")
        
        if st.button("🔄 Indexar (Sobreescribir)"):
            if not os.path.exists(data_folder):
                st.error(f"La carpeta '{data_folder}' no existe.")
            else:
                files = [f for f in os.listdir(data_folder) if f.endswith(".pdf")]
                if not files:
                    st.warning("No hay PDFs.")
                else:
                    bar = st.progress(0)
                    for i, f in enumerate(files):
                        run_indexing_service(os.path.join(data_folder, f), rag_service)
                        bar.progress((i+1)/len(files))
                    st.success("¡Indexado Completo!")
                    time.sleep(1)
                    st.rerun()

        st.divider()
        top_k = st.slider("Top-K Recuperados", 1, 10, 3)

    # --- TABS ---
    tab1, tab2 = st.tabs(["🔎 Consulta (Search)", "📊 Benchmark Ragas"])

    # --- TAB 1: BÚSQUEDA ---
    with tab1:
        st.subheader("Búsqueda Semántica")
        query = st.text_input("Escribe tu consulta:")

        if query:
            start_time = time.time()
            results = run_retrieval_service(query, rag_service, top_k=top_k)
            end_time = time.time()

            st.markdown(f"**Resultados:** {len(results)} chunks en {end_time - start_time:.3f}s")

            if not results:
                st.warning("No se encontraron resultados.")

            for i, chunk in enumerate(results, start=1):
                # Extraer Score de metadata
                score = 0.0
                if chunk.metadata and "score" in chunk.metadata:
                    score = chunk.metadata["score"]
                
                # Color del score
                score_color = "green" if score > 0.7 else "orange" if score > 0.5 else "red"

                with st.expander(f"Resultado #{i} | Score: :{score_color}[{score:.4f}]"):
                    st.markdown(f"**📄 Fuente:** `{chunk.source_file}` (Pág {chunk.page})")
                    st.markdown(f"**🆔 Chunk ID:** `{chunk.chunk_id}`")
                    st.info(chunk.content)
                    st.json(chunk.metadata, expanded=False)

    # --- TAB 2: BENCHMARK ---
    with tab2:
        st.subheader("Evaluación con Ragas (Local)")
        st.markdown(f"Usando dataset: `{DATASET_PATH}` y modelo local.")

        if st.button("🚀 Ejecutar Benchmark Real"):
            evaluator = BaselineEvaluator(rag_service)
            
            with st.spinner("Ejecutando evaluación..."):
                df_res, metrics_obj = evaluator.run_benchmark()
            
            if isinstance(df_res, str): # Manejo de errores
                st.error(df_res)
            else:
                st.success("¡Evaluación Completada!")
                
                # 1. Calcular Promedios (Solución al error de lista)
                faith_score = df_res['faithfulness'].mean()
                relevancy_score = df_res['answer_relevancy'].mean()

                col1, col2 = st.columns(2)
                col1.metric("Faithfulness", f"{faith_score:.4f}")
                col2.metric("Answer Relevancy", f"{relevancy_score:.4f}")

                # 2. Mostrar Tabla (Solución al error de columnas)
                # Definimos las columnas que Ragas suele devolver
                target_cols = ['user_input', 'faithfulness', 'answer_relevancy', 'response']
                
                # Seleccionamos solo las que existen en df_res para evitar KeyErrors
                final_cols = [c for c in target_cols if c in df_res.columns]
                
                st.dataframe(df_res[final_cols], use_container_width=True)
                
                # Opción de descargar
                csv = df_res.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "💾 Descargar Resultados CSV",
                    csv,
                    "ragas_results.csv",
                    "text/csv"
                )

if __name__ == "__main__":
    main()