FROM python:3.9-slim-buster

WORKDIR /app

COPY . /app/

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit","run","app.py"]