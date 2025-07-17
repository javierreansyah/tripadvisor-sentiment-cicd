from fastapi import APIRouter
from app.config import MODEL_SERVER_URL

router = APIRouter()

@router.get("/")
async def root():
    """Root endpoint providing a status message."""
    return {"message": f"FastAPI metrics exporter is running. Using model server at {MODEL_SERVER_URL}"}
