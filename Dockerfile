FROM python:3.9.6

RUN python -m pip install --upgrade pip

COPY . /app
WORKDIR /app
RUN python -m pip install .
RUN python -m gilda.resources
ENTRYPOINT gilda --port 8001 --host "0.0.0.0"
