import streamlit as st
import requests
import pandas as pd
import io
from dotenv import load_dotenv
import os

load_dotenv()
HOST = os.getenv("HOST")
PORT = os.getenv("PORT")

st.title("Resume Screener")

st.write("Загрузите резюме в формате PDF или ZIP.")

uploaded_file = st.file_uploader("Выберите файл", type=["pdf", "zip"])

if uploaded_file is not None:
    with st.spinner("Обрабатываем файл..."):
        if uploaded_file.type == "application/pdf":
            files = {"file": uploaded_file}
            response = requests.post(
                f"http://{HOST}:{PORT}/resume_analyzer/", files=files
            )
            result = response.json()

            if response.ok:
                st.subheader("Результаты для кандидата:")
                st.write(f"Имя: {result['Имя кандидата']}")
                st.write(f"Email: {result['Email']}")
                st.write(f"Телефон: {result['Номер телефона']}")
                st.write(f"Telegram: {result['Telegram']}")
                st.write(f"Предсказание: {result['Прогноз']}")
            else:
                st.error(response.text)

        elif uploaded_file.type in ["application/x-zip-compressed", "application/zip"]:
            files = {"file": uploaded_file}
            response = requests.post(
                f"http://{HOST}:{PORT}/resume_analyzer/", files=files
            )

            if response.ok:
                excel_file = io.BytesIO(response.content)
                df = pd.read_excel(excel_file)

                st.subheader("Результаты для кандидатов:")
                st.dataframe(df)

                st.download_button(
                    label="Скачать результаты",
                    data=excel_file,
                    file_name="resumes.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            else:
                st.error(response.text)
