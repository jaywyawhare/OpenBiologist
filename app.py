import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from src.services.mcp_server import create_mcp_server
from src.tools.mcp_tools import register_tools

TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")

assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"

mcp = create_mcp_server()

register_tools(mcp)

async def main():
    print("Starting OpenBiologist MCP server on http://0.0.0.0:8086")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())
