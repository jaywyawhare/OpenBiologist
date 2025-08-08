import os

def get_my_number() -> str:
    """Get the phone number from environment variables."""
    return os.environ.get("MY_NUMBER", "")

async def validate() -> str:
    """Validation tool required by Puch."""
    return get_my_number() 