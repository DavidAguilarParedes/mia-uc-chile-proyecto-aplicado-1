import streamlit as st
import os
from dotenv import load_dotenv
import nest_asyncio

# --- Importaciones RAG Core ---
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator

# --- Importaciones LangChain (Generación) ---
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage

# Configuración inicial
nest_asyncio.apply()
load_dotenv()

# ==========================================
# CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(
    page_title="Metabolomics AI Agent",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Agente de Anotación Metabolómica")
st.markdown("""
Este sistema utiliza **RAG Híbrido** para identificar features metabólicas basándose en 
masa exacta ($m/z$), tiempo de retención ($RT$) y literatura científica interna.
""")

# ==========================================
# BARRA LATERAL (INPUTS TÉCNICOS)
# ==========================================
with st.sidebar:
    st.header("🔬 Parámetros de la Feature")
    
    # Inputs numéricos clave
    target_mz = st.number_input(
        "Masa/Carga (m/z)", 
        value=449.107, 
        format="%.4f",
        help="Valor experimental del espectrómetro de masas."
    )
    
    tolerance = st.slider(
        "Tolerancia (Da)", 
        min_value=0.01, 
        max_value=1.0, 
        value=0.5,
        step=0.01,
        help="Ventana de búsqueda para el filtro de masa."
    )
    
    target_rt = st.number_input(
        "Tiempo de Retención (RT min)", 
        value=8.2, 
        format="%.2f",
        help="Opcional. Usado para contexto."
    )
    
    st.divider()
    st.caption("Conectado a: Qdrant Cloud ☁️")

# ==========================================
# LÓGICA DE CONEXIÓN (CACHED)
# ==========================================
@st.cache_resource
def init_rag_system():
    """Inicializa conexiones costosas una sola vez."""
    
    # 1. Validar Credenciales
    if not os.getenv("QDRANT_URL") or not os.getenv("OPENAI_API_KEY"):
        st.error("❌ Faltan credenciales en el archivo .env")
        st.stop()

    # 2. Configurar Embeddings (LlamaIndex)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    # 3. Conectar a Qdrant
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name="metabolomics_agent_db", # TU COLECCIÓN
        enable_hybrid=True
    )
    
    # Recuperar índice
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    # 4. Configurar LLM (LangChain)
    llm = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0)
    
    return index, client, llm

# Cargar sistema
try:
    index, qdrant_client, llm_chat = init_rag_system()
except Exception as e:
    st.error(f"Error conectando al sistema: {e}")
    st.stop()

# ==========================================
# INTERFAZ PRINCIPAL
# ==========================================

# Área de pregunta
query = st.text_area(
    "Consulta del Investigador:", 
    value="¿Qué compuesto es putativamente y qué actividades biológicas reportadas tiene?",
    height=100
)

# Botón de Acción
if st.button("🔍 Analizar Feature", type="primary"):
    
    if not query:
        st.warning("Por favor ingresa una pregunta.")
    else:
        with st.spinner("🔎 Buscando en base de datos vectorial y generando reporte..."):
            try:
                # ---------------------------------------------------------
                # PASO 1: RETRIEVAL (Filtro Numérico + Búsqueda Híbrida)
                # ---------------------------------------------------------
                
                # Definir filtros estrictos de m/z
                filters = MetadataFilters(
                    filters=[
                        MetadataFilter(key="mz_value", operator=FilterOperator.GTE, value=target_mz - tolerance),
                        MetadataFilter(key="mz_value", operator=FilterOperator.LTE, value=target_mz + tolerance),
                    ]
                )
                
                # Crear Retriever
                retriever = index.as_retriever(
                    filters=filters,
                    similarity_top_k=5, # Traemos top 5 chunks
                    vector_store_kwargs={"qdrant_client": qdrant_client}
                )
                
                # Ejecutar búsqueda
                results = retriever.retrieve(query)
                
                if not results:
                    st.warning(f"⚠️ No se encontraron documentos para m/z {target_mz} (+/- {tolerance}). Intenta aumentar la tolerancia.")
                else:
                    # ---------------------------------------------------------
                    # PASO 2: GENERACIÓN (LangChain)
                    # ---------------------------------------------------------
                    
                    # Preparar contexto para el prompt
                    context_str = ""
                    sources_data = [] # Para mostrar en la UI luego
                    
                    for r in results:
                        meta = r.metadata
                        # Guardar para visualización
                        sources_data.append({
                            "file": meta.get('file_name', 'Desconocido'),
                            "mz": meta.get('mz_value', 'N/A'),
                            "compound": meta.get('compound_name', 'Sin nombre'),
                            "snippet": r.text
                        })
                        # Guardar para el LLM
                        context_str += f"- Fuente: {meta.get('file_name')}\n"
                        context_str += f"- Compuesto: {meta.get('compound_name')} (m/z {meta.get('mz_value')})\n"
                        context_str += f"- Info: {r.text}\n\n"

                    # Prompt del Experto
                    system_prompt = """Eres un asistente experto en Química Analítica y Metabolómica. 
                    Genera un reporte técnico basado SOLO en el contexto proporcionado.
                    Estructura tu respuesta:
                    1. Identidad Putativa (basada en m/z).
                    2. Bioactividad reportada.
                    3. Referencias (qué documento dice qué).
                    """
                    
                    user_prompt = f"""
                    DATOS DE ENTRADA:
                    - Feature m/z: {target_mz}
                    - RT: {target_rt} min
                    - Consulta: {query}
                    
                    CONTEXTO RECUPERADO (QDRANT):
                    {context_str}
                    """
                    
                    # Generar respuesta
                    response = llm_chat.invoke([
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt)
                    ])
                    
                    # ---------------------------------------------------------
                    # PASO 3: VISUALIZACIÓN
                    # ---------------------------------------------------------
                    
                    st.success("✅ Análisis Completado")
                    
                    # Columna izquierda: Reporte
                    # Columna derecha: Datos clave
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader("📝 Informe de Anotación")
                        st.markdown(response.content)
                        
                    with col2:
                        st.subheader("📊 Datos Recuperados")
                        st.metric("Documentos Usados", len(results))
                        st.metric("Feature m/z Objetivo", target_mz)
                        
                        # Mostrar compuestos únicos encontrados
                        compuestos_unicos = set([s['compound'] for s in sources_data if s['compound']])
                        if compuestos_unicos:
                            st.info(f"**Candidatos:**\n" + "\n".join([f"- {c}" for c in compuestos_unicos]))

                    # Expander para ver fuentes (Transparencia)
                    with st.expander("📚 Ver Evidencia Documental (Fuentes Recuperadas)"):
                        for i, source in enumerate(sources_data):
                            st.markdown(f"**Documento {i+1}:** `{source['file']}`")
                            st.caption(f"m/z detectado: {source['mz']} | Compuesto: {source['compound']}")
                            st.text(source['snippet'][:300] + "...")
                            st.divider()

            except Exception as e:
                st.error(f"Ocurrió un error durante el análisis: {e}")

# Footer
st.markdown("---")
st.caption("Desarrollado para Proyecto Aplicado I - RAG en Metabolómica")