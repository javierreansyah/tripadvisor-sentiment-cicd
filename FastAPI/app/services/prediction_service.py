import asyncio
from .state import app_state

async def predict_sentiment(text: str) -> str:
    if app_state["model_cache"]["model"] is None:
        raise RuntimeError("No model is currently loaded. Please wait for model initialization.")
    
    def _predict_sync(text_to_predict):
        model = app_state["model_cache"]["model"]
        pred = model.predict([text_to_predict])[0]
        return pred
    
    prediction = await asyncio.to_thread(_predict_sync, text)
    
    return prediction
