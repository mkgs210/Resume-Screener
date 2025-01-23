from fastapi import APIRouter, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from tempfile import NamedTemporaryFile
import os
import pandas as pd
import uuid
from openai import OpenAIError
from starlette.requests import Request
from backend.services.resume_analyzer import ResumeAnalyzerService
from backend.exceptions import (
    EmptyPDFError,
    IncorrectZIPContentError,
    IncorrectFileType,
)


router = APIRouter()


def remove_file(path: str):
    os.remove(path)


@router.post("/")
async def analyze_resume(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    resume_analyzer_service = ResumeAnalyzerService()
    resume_analyzer_service.set_model(
        request.app.state.openai_client,
        request.app.state.prediction_model,
    )

    try:
        result = await resume_analyzer_service.analyze_resume(file)
    except EmptyPDFError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except IncorrectZIPContentError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except IncorrectFileType as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except OpenAIError as e:
        raise HTTPException(
            status_code=500, detail="The LLM is not available, try again later"
        ) from e

    if file.filename.endswith(".pdf"):
        return JSONResponse(content=result)

    elif file.filename.endswith(".zip"):
        df = pd.DataFrame(result)

        unique_filename = f"{uuid.uuid4()}.xlsx"

        with NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            df.to_excel(tmp.name, index=False)
            tmp_path = tmp.name

        unique_tmp_path = os.path.join(os.path.dirname(tmp_path), unique_filename)
        os.rename(tmp_path, unique_tmp_path)

        background_tasks.add_task(remove_file, unique_tmp_path)
        return FileResponse(
            unique_tmp_path,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=unique_filename,
        )
