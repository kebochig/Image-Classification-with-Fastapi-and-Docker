FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN mkdir /models

COPY my_model.h5 my_model.h5

COPY ./app.py /app

EXPOSE 8000

CMD ["python", "app.py"]