![PubChem MCP Server Logo](pubchem-mcp-server-logo.png)

# Proyecto Aplicado 1 - UC Chile

Este repositorio contiene dos proyectos principales:

1. **PubChem-MCP-Server**
2. **RAG / LangGraph Agent**

---

## 1. PubChem-MCP-Server

Servidor MCP no oficial para acceder a la base de datos química de PubChem. Proporciona acceso a más de 110 millones de compuestos químicos con propiedades moleculares, bioensayos y herramientas de química computacional.

### Características

#### 🔍 Chemical Search & Retrieval (6 herramientas)
- **search_compounds** - Buscar por nombre, número CAS, fórmula o identificador
- **get_compound_info** - Información detallada por CID
- **search_by_smiles** - Búsqueda exacta por SMILES
- **search_by_inchi** - Búsqueda por InChI/InChI key
- **search_by_cas_number** - Lookup por CAS
- **get_compound_synonyms** - Todos los nombres y sinónimos

#### 🧬 Structure Analysis & Similarity (5 herramientas)
- **search_similar_compounds** - Búsqueda por similitud Tanimoto
- **substructure_search** - Buscar subestructuras
- **superstructure_search** - Buscar compuestos mayores que contengan la consulta
- **get_3d_conformers** - Información estructural 3D
- **analyze_stereochemistry** - Análisis de quiralidad e isómeros

#### ⚗️ Chemical Properties & Descriptors (6 herramientas)
- **get_compound_properties** - Peso molecular, logP, TPSA, etc.
- **calculate_descriptors** - Descriptores moleculares completos
- **predict_admet_properties** - Predicción ADMET
- **assess_drug_likeness** - Lipinski Rule of Five
- **analyze_molecular_complexity** - Accesibilidad sintética
- **get_pharmacophore_features** - Mapas de farmacóforos

#### 🧪 Bioassay & Activity Data (5 herramientas)
- **search_bioassays** - Buscar ensayos biológicos
- **get_assay_info** - Protocolos detallados
- **get_compound_bioactivities** - Datos de actividad de compuestos
- **search_by_target** - Buscar compuestos por objetivo
- **compare_activity_profiles** - Comparación entre compuestos

#### ⚠️ Safety & Toxicity (4 herramientas)
- **get_safety_data** - Clasificaciones de riesgo GHS
- **get_toxicity_info** - LD50, carcinogenicidad
- **assess_environmental_fate** - Biodegradación
- **get_regulatory_info** - Regulaciones FDA/EPA

#### 🔗 Cross-References & Integration (4 herramientas)
- **get_external_references** - Enlaces a ChEMBL, DrugBank, etc.
- **search_patents** - Información de patentes químicas
- **get_literature_references** - Citaciones PubMed
- **batch_compound_lookup** - Procesamiento masivo (hasta 200 compuestos)

### Plantillas de recursos

- `pubchem://compound/{cid}`
- `pubchem://structure/{cid}`
- `pubchem://properties/{cid}`
- `pubchem://bioassay/{aid}`
- `pubchem://similarity/{smiles}`
- `pubchem://safety/{cid}`

### Instalación

```bash
cd PubChem-MCP-Server
npm install
npm run build
npm start
