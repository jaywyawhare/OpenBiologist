import asyncio
import os
from dotenv import load_dotenv
from src.services.mcp_server import create_mcp_server
from src.services.validation import validate
from src.tools.job_finder import job_finder, JobFinderDescription
from src.tools.image_processor import make_img_black_and_white, MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION

# --- Load environment variables ---
load_dotenv()

TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")

assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"

# --- Create MCP Server ---
mcp = create_mcp_server()

# --- Register Tools ---
@mcp.tool
async def validate_tool() -> str:
    return await validate()

@mcp.tool(description=JobFinderDescription.model_dump_json())
async def job_finder_tool(
    user_goal: str,
    job_description: str | None = None,
    job_url: str | None = None,
    raw: bool = False,
) -> str:
    return await job_finder(user_goal, job_description, job_url, raw)

@mcp.tool(description=MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION.model_dump_json())
async def make_img_black_and_white_tool(
    puch_image_data: str = None,
):
    return await make_img_black_and_white(puch_image_data)

# --- Run MCP Server ---
async def main():
    print("🚀 Starting OpenBiologist MCP server on http://0.0.0.0:8086")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())
