FROM python:3.11-slim

WORKDIR /app

COPY . /app/frontend

RUN pip install -i https://art.x5.ru/artifactory/api/pypi/pypi/simple --trusted-host art.x5.ru --no-cache-dir -r frontend/requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]