from langchain_mcp_adapters.client import MultiServerMCPClient
import nest_asyncio
import asyncio

nest_asyncio.apply()

# Connect to the mcp-time server for timezone-aware operations
# This Go-based server provides tools for current time, relative time parsing,
# timezone conversion, duration arithmetic, and time comparison
mcp_client = MultiServerMCPClient(
    {
        "time": {
            "transport": "stdio",
            "command": "node",
            "args": ["/home/david/uc_chile/proyecto_aplicado_1/PubChem-MCP-Server/build/index.js"],
        }
    },
)

# Load tools from the MCP server
async def _load_tools():
    return await mcp_client.get_tools()

mcp_tools = asyncio.run(_load_tools())

print(f"Loaded {len(mcp_tools)} MCP tools: {[t.name for t in mcp_tools]}")


from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-5",
    tools=mcp_tools,
    system_prompt="You are a helpful assistant",
)


