from src.services.validation import validate
from src.tools.job_finder import job_finder, JobFinderDescription
from src.tools.image_processor import make_img_black_and_white, MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION
from src.tools.protein_predictor import predict_protein_structure, check_protein_prediction_status, ProteinStructurePredictionDescription
from src.tools.bio_database_search import search_protein_database, BioDatabaseSearchDescription


def register_tools(mcp):
    """Register all MCP tools with the provided MCP server instance."""
    
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

    @mcp.tool(description=ProteinStructurePredictionDescription.model_dump_json())
    async def predict_protein_structure_tool(
        sequence: str,
        job_name: str = "Protein Structure Prediction",
        wait_for_completion: bool = False,
    ) -> str:
        return await predict_protein_structure(sequence, job_name, wait_for_completion)

    @mcp.tool(description="Check the status of a protein structure prediction job")
    async def check_protein_prediction_status_tool(
        job_id: str,
    ) -> str:
        return await check_protein_prediction_status(job_id)

    @mcp.tool(description=BioDatabaseSearchDescription.model_dump_json())
    async def search_protein_database_tool(
        query: str,
        database: str = "all",
        max_results: int = 5,
    ) -> str:
        return await search_protein_database(query, database, max_results)
