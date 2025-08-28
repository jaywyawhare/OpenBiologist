import os
from fastmcp import FastMCP
from src.auth.bearer_provider import SimpleBearerAuthProvider

# MCP server for OpenBiologist with local protein folding capabilities

def create_mcp_server() -> FastMCP:
    """Create and configure the MCP server."""
    token = os.environ.get("AUTH_TOKEN")
    if not token:
        raise ValueError("Please set AUTH_TOKEN in your .env file")
    
    return FastMCP(
        "OpenBiologist MCP Server",
        auth=SimpleBearerAuthProvider(token),
    ) 