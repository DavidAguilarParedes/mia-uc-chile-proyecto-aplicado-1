# Proyecto Aplicado 1 - UC Chile

Este repositorio contiene dos subproyectos principales:

1. PubChem-MCP-Server
2. RAG / LangGraph Agent

---

## 1. PubChem-MCP-Server

El servidor completo se encuentra en la carpeta `PubChem-MCP-Server`.
Para instrucciones detalladas de instalación, uso y ejemplos, revisa el README dentro de esa carpeta.

```bash
cd PubChem-MCP-Server
npm install
npm run build
npm start
```

---

## 2. RAG / LangGraph Agent

El servidor RAG se encuentra en la carpeta `rag`.
Para levantar el servidor:

```bash
cd rag
uv sync       # Sincroniza los datos y herramientas
langgraph dev # Levanta el servidor LangGraph
```

---

## Notas importantes

* Configura tus variables de entorno en `.env` dentro de cada subcarpeta según corresponda.
* Los archivos `.env` no deben subirse al repositorio (están en `.gitignore`).
* Asegúrate de tener instaladas las dependencias de Node.js para PubChem y `uv` + `langgraph` para RAG.

---

## Licencia

MIT License - Ver archivo LICENSE.
