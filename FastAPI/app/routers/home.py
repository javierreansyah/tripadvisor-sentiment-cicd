import asyncio
from fastapi import APIRouter, Request, BackgroundTasks, HTTPException, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.services import (
    get_dashboard_data, 
    run_retraining_pipeline,
    promote_model_to_prod,
    app_state,
    generate_and_save_gemini_data,
    predict_sentiment
)

router = APIRouter()
templates = Jinja2Templates(directory="templates")

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    prediction: int

@router.get("/", response_class=HTMLResponse)
async def show_home_dashboard(request: Request):
    dashboard_data = await get_dashboard_data()
    return templates.TemplateResponse("home.html", {"request": request, **dashboard_data})

@router.post("/retrain")
async def trigger_retraining(background_tasks: BackgroundTasks):
    """Triggers the full retraining pipeline as a background task."""
    if app_state["retraining"]["is_training"]:
        raise HTTPException(status_code=409, detail="A retraining job is already in progress.")
    background_tasks.add_task(run_retraining_pipeline)
    
    return RedirectResponse(url="/", status_code=303)

@router.post("/promote/{version}")
async def promote_model(version: int):
    await promote_model_to_prod(version)
    return RedirectResponse(url="/", status_code=303)

@router.post("/generate")
async def handle_generate_data(style: str = Form(None), quantity: int = Form(...)):
    try:
        await asyncio.to_thread(generate_and_save_gemini_data, style, quantity)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return RedirectResponse(url="/", status_code=303)

@router.post("/predict", response_model=PredictResponse)
async def predict_text_sentiment(request: PredictRequest):
    try:
        prediction = await predict_sentiment(request.text)
        prediction_int = 1 if prediction.lower() == 'positive' else 0
        return PredictResponse(prediction=prediction_int)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
