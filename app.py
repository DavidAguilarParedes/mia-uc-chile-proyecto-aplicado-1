import os
import time
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# --- IMPORTS DE TU ARQUITECTURA HEXAGONAL ---
from src.infrastructure.embeddings.huggingface import HuggingFaceEmbedder
from src.infrastructure.vector_stores.qdrant_db import QdrantImpl
from src.application.services.rag_service import VectorStoreService, run_indexing_service, run_retrieval_service

# Cargar variables de entorno
load_dotenv()

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA
# ==============================================================================
st.set_page_config(page_title="Hito 1: Clean RAG Architecture", layout="wide")

# ==============================================================================
# 1. CAPA DE APLICACIÓN (Inicialización)
# ==============================================================================

@st.cache_resource
def get_vector_service() -> VectorStoreService:
    """
    Instancia los adaptadores y el servicio de aplicación.
    Al usar cache_resource, la DB en memoria de Qdrant no se borra al interactuar con la UI.
    """
    # 1. Adaptador de Embeddings (puedes cambiarlo por GeminiEmbedder si prefieres)
    embedder = HuggingFaceEmbedder(model_name="all-MiniLM-L6-v2")
    
    # 2. Adaptador de Base de Datos Vectorial
    db_impl = QdrantImpl(collection_name="rag_chunks")
    
    # 3. Inyección de dependencias en el Servicio
    service = VectorStoreService(embedder=embedder, db_impl=db_impl)
    
    return service

# ==============================================================================
# 2. SISTEMA DE BENCHMARK (Hardcoded / Mock)
# ==============================================================================

class BaselineEvaluator:
    """
    Mock del evaluador para cumplir con el requisito de dejarlo 'hardcoded'.
    """
    def run_benchmark(self):
        # Simulamos una espera
        time.sleep(1)
        
        # Datos Hardcoded
        data = [
            {
                "Query": "What is the m/z of 4-Dihydroxyacetophenone?",
                "Expected": "153.0546",
                "Hit (Encontrado)": "✅",
                "Top Context": "The m/z value observed was 153.0546 corresponding to..."
            },
            {
                "Query": "Lipid composition changes in cocoa?",
                "Expected": "alkalization",
                "Hit (Encontrado)": "✅",
                "Top Context": "Alkalization significantly alters the lipid profile..."
            }
        ]
        
        df = pd.DataFrame(data)
        accuracy = 1.0 # 100% fake accuracy
        return df, accuracy

# ==============================================================================
# 3. INTERFAZ GRÁFICA (Streamlit)
# ==============================================================================

def main():
    st.title("🧪 Hito 1: Clean Architecture RAG")
    st.markdown("Implementación desacoplada usando Hexagonal Architecture.")

    # Obtener el servicio (Singleton)
    rag_service = get_vector_service()

    # --- SIDEBAR: GESTIÓN DE DATOS ---
    with st.sidebar:
        st.header("⚙️ Ingesta de Datos")
        
        data_folder = "data"
        
        if st.button("🔄 Indexar Archivos en 'data/'"):
            if not os.path.exists(data_folder):
                st.error(f"La carpeta '{data_folder}' no existe.")
            else:
                files = os.listdir(data_folder)
                files = [f for f in files if f.endswith(".pdf")]
                
                if not files:
                    st.warning("No hay PDFs en la carpeta data.")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, filename in enumerate(files):
                        status_text.text(f"Procesando: {filename}...")
                        full_path = os.path.join(data_folder, filename)
                        
                        # --- LLAMADA AL SERVICIO DE INDEXADO ---
                        try:
                            run_indexing_service(full_path, rag_service)
                        except Exception as e:
                            st.error(f"Error en {filename}: {e}")
                            
                        progress_bar.progress((i + 1) / len(files))
                    
                    status_text.success("¡Indexado completado!")
                    time.sleep(1)
                    status_text.empty()
                    progress_bar.empty()

        st.divider()
        top_k = st.slider("Documentos a recuperar (Top-K)", 1, 10, 3)

    # --- TABS PRINCIPALES ---
    tab1, tab2 = st.tabs(["🔎 Consulta (Search)", "📊 Métricas (Benchmark)"])

    # --- TAB 1: BÚSQUEDA ---
    with tab1:
        st.subheader("Búsqueda Semántica")
        query = st.text_input("Escribe tu consulta:")

        if query:
            start_time = time.time()
            
            # --- LLAMADA AL SERVICIO DE RETRIEVAL ---
            # Devuelve una lista de objetos ProcessedChunk
            results = run_retrieval_service(query, rag_service, top_k=top_k)
            
            end_time = time.time()

            st.markdown(f"**Resultados encontrados** ({end_time - start_time:.4f} seg):")

            if not results:
                st.info("No se encontraron resultados. ¿Has indexado los documentos?")

            for i, chunk in enumerate(results, start=1):
                # chunk es una instancia de ProcessedChunk
                score = chunk.metadata.get("score", 0.0) if chunk.metadata else 0.0
                
                with st.expander(f"Resultado #{i} (Score aprox: {score:.4f})"):
                    st.markdown(f"**🆔 Chunk ID:** `{chunk.chunk_id}`")
                    st.markdown(f"**📄 Archivo:** `{chunk.source_file}` | **Pág:** `{chunk.page}`")
                    st.info(chunk.content)
                    
                    # Mostrar metadatos extra si existen
                    if chunk.metadata:
                        st.json(chunk.metadata, expanded=False)

    # --- TAB 2: BENCHMARK ---
    with tab2:
        st.subheader("Evaluación (Simulada)")
        st.markdown("Prueba de integración con Golden Dataset.")

        evaluator = BaselineEvaluator()

        if st.button("🚀 Ejecutar Benchmark"):
            with st.spinner("Evaluando..."):
                df_results, accuracy = evaluator.run_benchmark()

            col1, col2 = st.columns(2)
            col1.metric(label="Accuracy", value=f"{accuracy * 100:.1f}%")
            col2.metric(label="Total Queries", value=len(df_results))

            st.dataframe(df_results, use_container_width=True)

if __name__ == "__main__":
    main()