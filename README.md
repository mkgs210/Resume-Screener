# resume_screener

## Структура проекта

```plain
├── backend/
│   ├── prediction_model/
│   │   └── prediction_model.py
│   ├── routers/
│   │   └── resume_analyzer.py
│   ├── services/
│   │   └── resume_analyzer.py
│   ├── Dockerfile
│   ├── exceptions.py
│   ├── main.py
│   └── requirements.txt
├── data/
│   ├── preprocessed/
│   │   ├── .gitkeep
│   │   └── ...
|   └── raw/
│       ├── .gitkeep
|       └── ...
├── frontend/
│   ├── app.py
│   ├── Dockerfile
|   └── requirements.txt
├── models/
│   └── trained_model/
        └── ...
├── notebooks/
│   └── ...
├── tests/
│   └── ...
├── .dockerignore
├── .env
├── .flake8
├── .gitignore
├── docker-compose.yml
└── README.md
```

* backend - директория, содержащая код бэкенда на FastAPI
    * prediction_model - директория, содержащая код для работы с обученной моделью
    * routers - директория для хранения доменных роутеров и эндпоинтов
    * services - директория для хранения сервисов с бизнес-логикой, вызывающих репозитории или ML-модели
    * main.py - точка входа в сервис
* data - директория для хранения данных
* frontend - директория, содержащая код UI на Streamlit
    * app.py - основной файл с кодом для создания UI
* models - директория для хранения обученных моделей классификации резюме
* notebooks - директория для хранения ноутбуков с экспериментами
* tests - директория с тестами

## Разработка

```
git clone https://scm.x5.ru/nonproduct/resume_screener.git
cd resume_screener
python -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

## Команды

### Развертывание сервиса через докер

Для развертывания приложения необходимо выполнить следующую команду:

```
docker-compose up -d
```

### Запуск сервиса локально

Для запуска бэкенда необходимо в терминале из корневой директории репозитория выполнить следующую команду:

```
uvicorn backend.main:app --reload
```

После запуска команды с сервисом можно работать через интерфейс, предоставляемый FastAPI. Для этого необходимо в браузере открыть ссылку `http://localhost:8000/docs`.

Чтобы работать с сервисом через интерфейс Streamlit, необходимо открыть еще один терминал и из корневой директории репозитория выполнить следуюую команду:

```
streamlit run frontend/app.py
```

После этого необходимо в браузере открыть ссылку `http://localhost:8501`.

### Использование сервиса, развернутого на ноде

Если сервис уже развернут на ноде через докер, то для работы с ним через интерфейс Streamlit необходимо открыть в браузере ссылку `http://{имя ноды}:8501`. Например, если сервис развернут на ноде mn-hdap64, то для его использования необходимо открыть в браузере ссылку `http://mn-hdap64:8501`.