import pytest
from faker import Faker
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dotenv import load_dotenv
import os
from pathlib import Path
import openai
from starlette.datastructures import UploadFile as StarletteUploadFile
import io
from fastapi.testclient import TestClient
from backend.prediction_model.prediction_model import ResumeBasedPredictionModel
from backend.main import app
from backend.services.resume_analyzer import ResumeAnalyzerService
from backend.exceptions import (
    EmptyPDFError,
    IncorrectZIPContentError,
    IncorrectFileType,
)

load_dotenv()
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
BASE_PREDICTION_MODEL_PATH = os.getenv("BASE_PREDICTION_MODEL_PATH")
TRAINED_PREDICTION_MODEL_PATH = os.getenv("TRAINED_PREDICTION_MODEL_PATH")

BASE_PREDICTION_MODEL_PATH = (
    Path(__file__).parent.parent.parent / BASE_PREDICTION_MODEL_PATH
)
TRAINED_PREDICTION_MODEL_PATH = (
    Path(__file__).parent.parent.parent / TRAINED_PREDICTION_MODEL_PATH
)

client = TestClient(app)
faker = Faker()


@pytest.fixture(scope="module")
def get_resume_analyzer():
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

    openai_client = openai.AsyncClient(
        api_key="any-key",
        base_url=LLM_BASE_URL,
    )

    resume_analyzer = ResumeAnalyzerService()
    resume_analyzer.set_model(openai_client, prediction_model)

    return resume_analyzer


def test_set_model(get_resume_analyzer):
    resume_analyzer = get_resume_analyzer

    assert resume_analyzer.openai_client is not None
    assert resume_analyzer.prediction_model is not None


@pytest.mark.parametrize(
    "text, email",
    [
        ("abc.xyz@alumni.nu.edu.kz", "abc.xyz@alumni.nu.edu.kz"),
        ("abc123@yandex.ru", "abc123@yandex.ru"),
        ("abc@gmail.com", "abc@gmail.com"),
        ("abc.xy@phystech.edu", "abc.xy@phystech.edu"),
        ("abc@edu.hse.ru", "abc@edu.hse.ru"),
        ("abc@nes.ru", "abc@nes.ru"),
        ("abc@rambler.ru", "abc@rambler.ru"),
        ("abc@bk.ru", "abc@bk.ru"),
        ("1123 dfg abc@gmail.com adfg 12", "abc@gmail.com"),
        ("abc 123 xyz456 v r h", None),
    ],
)
def test_extract_email_from_text(get_resume_analyzer, text, email):
    resume_analyzer = get_resume_analyzer

    assert resume_analyzer._extract_email_from_text(text) == email


@pytest.mark.parametrize(
    "text, phone_number",
    [
        ("71234567890", "71234567890"),
        ("+71234567890", "71234567890"),
        ("+7-123-456-78-90", "71234567890"),
        ("+7 123 456 78 90", "71234567890"),
        ("+7 (123) 4567890", "71234567890"),
        ("+7(123)-456-7890", "71234567890"),
        ("+7 (123) 456-7890", "71234567890"),
        ("+7(123)4567890", "71234567890"),
        ("7 123 456 7890", "71234567890"),
        ("+7(123)456-78-90", "71234567890"),
        ("81234567890", "81234567890"),
        ("81234567890", "81234567890"),
        ("8-123-456-78-90", "81234567890"),
        ("8 123 456 78 90", "81234567890"),
        ("8 (123) 4567890", "81234567890"),
        ("8(123)-456-7890", "81234567890"),
        ("8 (123) 456-7890", "81234567890"),
        ("8(123)4567890", "81234567890"),
        ("8 123 456 7890", "81234567890"),
        ("8(123)456-78-90", "81234567890"),
        ("asdff наппапр 71234567890 fgf папввп", "71234567890"),
        ("абв abc 4424 fglh пхцеь", None),
    ],
)
def test_extract_phone_number_from_text(get_resume_analyzer, text, phone_number):
    resume_analyzer = get_resume_analyzer

    assert resume_analyzer._extract_phone_number_from_text(text) == phone_number


@pytest.mark.parametrize(
    "text, tg",
    [
        ("https://t.me/abc123", "https://t.me/abc123"),
        ("http://t.me/abc123", "http://t.me/abc123"),
        ("tg: http://t.me/abc123", "http://t.me/abc123"),
        ("Telegram: https://t.me/abc123", "https://t.me/abc123"),
    ],
)
@pytest.mark.asyncio
async def test_extract_tg_from_text(get_resume_analyzer, text, tg):
    resume_analyzer = get_resume_analyzer

    result = await resume_analyzer._extract_tg_from_text(text)

    assert result == tg


def test_extract_text_from_correct_pdf(get_resume_analyzer):
    resume_analyzer = get_resume_analyzer

    with open("tests/test_services/test_data/pdf_example_1.pdf", "rb") as f:
        file_content = f.read()

        file_like_object = io.BytesIO(file_content)

        upload_file = StarletteUploadFile(
            filename="test_file.pdf", file=file_like_object
        )

    result = resume_analyzer._extract_text_from_pdf(upload_file)

    assert result != ""


def test_extract_text_from_incorrect_pdf(get_resume_analyzer):
    resume_analyzer = get_resume_analyzer

    with open("tests/test_services/test_data/picture_without_text.pdf", "rb") as f:
        file_content = f.read()

        file_like_object = io.BytesIO(file_content)

        upload_file = StarletteUploadFile(
            filename="test_file.pdf", file=file_like_object
        )

    result = resume_analyzer._extract_text_from_pdf(upload_file)

    assert result == ""


@pytest.mark.parametrize(
    "text, name",
    [
        ("Иванов Иван Иванович", "Иванов Иван Иванович"),
        ("Петрова Анна Петровна", "Петрова Анна Петровна"),
        ("Иванов Иван", "Иванов Иван"),
        ("Петрова Анна", "Петрова Анна"),
        (
            "Иванов Иван Иванович\nМужчина, 36 лет, родился 12 апреля 1986",
            "Иванов Иван Иванович",
        ),
        (
            "Петрова Анна Петровна\nЖенщина, 36 лет, родилась 12 апреля 1986",
            "Петрова Анна Петровна",
        ),
    ],
)
@pytest.mark.asyncio
async def test_extract_name_from_text(get_resume_analyzer, text, name):
    resume_analyzer = get_resume_analyzer

    result = await resume_analyzer._extract_name_from_text(text)

    assert result == name


@pytest.mark.asyncio
async def test_extract_pd_from_text(get_resume_analyzer):
    resume_analyzer = get_resume_analyzer

    text = "Иванов Иван AbcXyz@gmail.com +7 (123) 4567890 https://t.me/abc123"

    result = await resume_analyzer._extract_pd_from_text(text)

    assert result != {}


@pytest.mark.parametrize(
    "file_path",
    [
        ("tests/test_services/test_data/pdf_example_1.pdf"),
        ("tests/test_services/test_data/pdf_example_2.pdf"),
    ],
)
@pytest.mark.asyncio
async def test_process_correct_pdf(get_resume_analyzer, file_path):
    resume_analyzer = get_resume_analyzer

    with open(file_path, "rb") as f:
        file_content = f.read()

        file_like_object = io.BytesIO(file_content)

        upload_file = StarletteUploadFile(
            filename="test_file.pdf", file=file_like_object
        )

    result = await resume_analyzer._process_pdf(upload_file)

    assert result != {}


@pytest.mark.asyncio
async def test_process_incorrect_pdf(get_resume_analyzer):
    resume_analyzer = get_resume_analyzer

    with open("tests/test_services/test_data/picture_without_text.pdf", "rb") as f:
        file_content = f.read()

        file_like_object = io.BytesIO(file_content)

        upload_file = StarletteUploadFile(
            filename="test_file.pdf", file=file_like_object
        )

    with pytest.raises(EmptyPDFError, match="The file must contain text!"):
        await resume_analyzer._process_pdf(upload_file)


@pytest.mark.asyncio
async def test_process_correct_zip(get_resume_analyzer):
    resume_analyzer = get_resume_analyzer

    with open("tests/test_services/test_data/zip_with_pdf.zip", "rb") as f:
        file_content = f.read()

    file_object = io.BytesIO(file_content)

    upload_file = StarletteUploadFile(filename="test_file.zip", file=file_object)

    result = await resume_analyzer._process_zip(upload_file)

    assert result != {
        "Имя файла": [],
        "Имя кандидата": [],
        "Email": [],
        "Номер телефона": [],
        "Telegram": [],
        "Прогноз": [],
    }


@pytest.mark.asyncio
async def test_process_correct_zip_from_macos(get_resume_analyzer):
    resume_analyzer = get_resume_analyzer

    with open("tests/test_services/test_data/zip_from_macos.zip", "rb") as f:
        file_content = f.read()

    file_object = io.BytesIO(file_content)

    upload_file = StarletteUploadFile(filename="test_file.zip", file=file_object)

    result = await resume_analyzer._process_zip(upload_file)

    assert result != {
        "Имя файла": [],
        "Имя кандидата": [],
        "Email": [],
        "Номер телефона": [],
        "Telegram": [],
        "Прогноз": [],
    }


@pytest.mark.asyncio
async def test_process_incorrect_zip_without_pdf(get_resume_analyzer):
    resume_analyzer = get_resume_analyzer

    with open("tests/test_services/test_data/zip_without_pdf.zip", "rb") as f:
        file_content = f.read()

    file_object = io.BytesIO(file_content)

    upload_file = StarletteUploadFile(filename="test_file.zip", file=file_object)

    with pytest.raises(
        IncorrectZIPContentError,
        match="ZIP contains not only files with the .pdf extension!",
    ):
        await resume_analyzer._process_zip(upload_file)


@pytest.mark.asyncio
async def test_process_incorrect_zip_with_doc_and_pdf(get_resume_analyzer):
    resume_analyzer = get_resume_analyzer

    with open("tests/test_services/test_data/zip_with_doc_and_pdf.zip", "rb") as f:
        file_content = f.read()

    file_object = io.BytesIO(file_content)

    upload_file = StarletteUploadFile(filename="test_file.zip", file=file_object)

    with pytest.raises(
        IncorrectZIPContentError,
        match="ZIP contains not only files with the .pdf extension!",
    ):
        await resume_analyzer._process_zip(upload_file)


@pytest.mark.asyncio
async def test_analyze_resume_incorrect_file_type(get_resume_analyzer):
    resume_analyzer = get_resume_analyzer

    with open("tests/test_services/test_data/doc_example_1.doc", "rb") as f:
        file_content = f.read()

    file_object = io.BytesIO(file_content)

    upload_file = StarletteUploadFile(filename="test_file.doc", file=file_object)

    with pytest.raises(
        IncorrectFileType,
        match="The file type must be pdf or zip!",
    ):
        await resume_analyzer.analyze_resume(upload_file)


@pytest.mark.asyncio
async def test_analyze_resume_correct_pdf(get_resume_analyzer):
    resume_analyzer = get_resume_analyzer

    with open("tests/test_services/test_data/pdf_example_1.pdf", "rb") as f:
        file_content = f.read()

    file_object = io.BytesIO(file_content)

    upload_file = StarletteUploadFile(filename="test_file.pdf", file=file_object)

    result = await resume_analyzer.analyze_resume(upload_file)

    assert result != {}


@pytest.mark.asyncio
async def test_analyze_resume_correct_zip(get_resume_analyzer):
    resume_analyzer = get_resume_analyzer

    with open("tests/test_services/test_data/zip_with_pdf.zip", "rb") as f:
        file_content = f.read()

    file_object = io.BytesIO(file_content)

    upload_file = StarletteUploadFile(filename="test_file.zip", file=file_object)

    result = await resume_analyzer.analyze_resume(upload_file)

    assert result != {
        "Имя файла": [],
        "Имя кандидата": [],
        "Email": [],
        "Номер телефона": [],
        "Telegram": [],
        "Прогноз": [],
    }
