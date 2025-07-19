import asyncio
from fastapi import APIRouter, Request, BackgroundTasks, HTTPException, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.services import (
    get_dashboard_data, 
    run_retraining_pipeline,
    promote_model_to_prod,
    app_state,
    generate_and_save_gemini_data
)

router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.get("/", response_class=HTMLResponse)
async def show_home_dashboard(request: Request):
    """Renders the combined home dashboard with retraining controls and data addition forms."""
    dashboard_data = await get_dashboard_data()
    return templates.TemplateResponse("home.html", {"request": request, **dashboard_data})

@router.post("/retrain")
async def trigger_retraining(background_tasks: BackgroundTasks):
    """Triggers the full retraining pipeline as a background task."""
    if app_state["retraining"]["is_training"]:
        raise HTTPException(status_code=409, detail="A retraining job is already in progress.")
    
    # Start the pipeline in the background so the UI doesn't hang
    background_tasks.add_task(run_retraining_pipeline)
    
    return RedirectResponse(url="/", status_code=303)

@router.post("/promote/{version}")
async def promote_model(version: int):
    """Sets the specified model version as the production model."""
    await promote_model_to_prod(version)
    return RedirectResponse(url="/", status_code=303)

@router.post("/generate")
async def handle_generate_data(style: str = Form(None), quantity: int = Form(...)):
    """
    Handles the Gemini data generation request by calling the generation service.
    """
    try:
        # The service function already handles API calls and file writing.
        # We run it in a thread because the Gemini API call is synchronous.
        await asyncio.to_thread(generate_and_save_gemini_data, style, quantity)
    except Exception as e:
        # Catch exceptions from the service and show an error
        raise HTTPException(status_code=500, detail=str(e))

    return RedirectResponse(url="/", status_code=303)
