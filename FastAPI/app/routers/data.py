import asyncio
from fastapi import APIRouter, Form, Request, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

# Import the new service functions
from app.services import add_manual_data, generate_and_save_gemini_data

router = APIRouter()

# Point to the 'templates' directory
templates = Jinja2Templates(directory="templates")

@router.get("/new", response_class=HTMLResponse)
async def show_new_data_form(request: Request):
    """
    Renders the data form from an external HTML file.
    """
    return templates.TemplateResponse("data_form.html", {"request": request})

@router.post("/new")
async def handle_add_new_data(review: str = Form(...), sentiment: int = Form(...)):
    """
    Handles the manual data submission by calling the data service.
    """
    # Run the synchronous file-writing operation in a separate thread
    await asyncio.to_thread(add_manual_data, review, sentiment)
    return RedirectResponse(url="/new", status_code=303)

@router.post("/generate")
async def handle_generate_data(style: str = Form(...), quantity: int = Form(...)):
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

    return RedirectResponse(url="/new", status_code=303)
