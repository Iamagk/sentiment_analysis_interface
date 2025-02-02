from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from model_interface import run_model
from rfsent import load_and_prepare_data
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    model: str
    sentiment_files: List[str]

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        logger.info(f"Received prediction request: {request}")

        # Validate input data
        if not request.ticker or not request.start_date or not request.end_date:
            raise ValueError("Ticker, start_date, and end_date are required")

        if not request.sentiment_files:
            raise ValueError("At least one sentiment file is required")

        data = load_and_prepare_data(request.ticker, request.start_date, request.end_date, request.sentiment_files)
        data.update({
            "ticker": request.ticker,
            "start_date": request.start_date,
            "end_date": request.end_date
        })
        result = run_model(request.model, data)
        return result
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
