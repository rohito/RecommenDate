FROM python:3.8.6-buster

COPY api /api
COPY RecommenDate /RecommenDate
COPY requirements.txt /requirements.txt
COPY rohit-creds.json /credentials.json
RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
