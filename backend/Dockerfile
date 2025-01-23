FROM python:3.11-slim

WORKDIR /app

COPY . /app/backend

RUN pip install -i https://art.x5.ru/artifactory/api/pypi/pypi/simple --trusted-host art.x5.ru --no-cache-dir -r backend/requirements.txt

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload"]