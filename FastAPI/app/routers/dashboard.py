from fastapi import APIRouter, Request, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.services import (
    get_dashboard_data, 
    run_retraining_pipeline,
    promote_model_to_prod,
    app_state # Corrected: Import the main app_state dictionary
)

router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.get("/dashboard", response_class=HTMLResponse)
async def show_dashboard(request: Request):
    """Renders the main dashboard UI."""
    dashboard_data = await get_dashboard_data()
    return templates.TemplateResponse("dashboard.html", {"request": request, **dashboard_data})

@router.post("/retrain")
async def trigger_retraining(background_tasks: BackgroundTasks):
    """Triggers the full retraining pipeline as a background task."""
    # Corrected: Check the status from within the app_state dictionary
    if app_state["retraining"]["is_training"]:
        raise HTTPException(status_code=409, detail="A retraining job is already in progress.")
    
    # Start the pipeline in the background so the UI doesn't hang
    background_tasks.add_task(run_retraining_pipeline)
    
    return RedirectResponse(url="/dashboard", status_code=303)

@router.post("/promote/{version}")
async def promote_model(version: int):
    """Sets the specified model version as the production model."""
    await promote_model_to_prod(version)
    return RedirectResponse(url="/dashboard", status_code=303)
