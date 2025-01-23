from fastapi import FastAPI
import openai
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
from backend.routers.resume_analyzer import router as resume_analyzer_router
from backend.prediction_model.prediction_model import ResumeBasedPredictionModel

load_dotenv()
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
BASE_PREDICTION_MODEL_PATH = os.getenv("BASE_PREDICTION_MODEL_PATH")
TRAINED_PREDICTION_MODEL_PATH = os.getenv("TRAINED_PREDICTION_MODEL_PATH")

BASE_PREDICTION_MODEL_PATH = Path(__file__).parent.parent / BASE_PREDICTION_MODEL_PATH
TRAINED_PREDICTION_MODEL_PATH = (
    Path(__file__).parent.parent / TRAINED_PREDICTION_MODEL_PATH
)


@asynccontextmanager
async def lifespan(app: FastAPI):

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_PREDICTION_MODEL_PATH,
        use_safetensors=True,
        device_map="cpu",
    )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_PREDICTION_MODEL_PATH,
        num_labels=2,
        device_map="cpu",
    )

    prediction_model = None
    if os.path.exists(TRAINED_PREDICTION_MODEL_PATH):
        prediction_model = ResumeBasedPredictionModel().load_model(
            base_model, tokenizer, TRAINED_PREDICTION_MODEL_PATH
        )

    app.state.prediction_model = prediction_model

    app.state.openai_client = openai.AsyncClient(
        api_key="any-key",
        base_url=LLM_BASE_URL,
    )

    yield


app = FastAPI(lifespan=lifespan)
app.include_router(resume_analyzer_router, prefix="/resume_analyzer")
