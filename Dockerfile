FROM python:3.9.6

RUN python -m pip install --upgrade pip
RUN python -m pip install gilda
ENTRYPOINT gilda --port 8001 --host "0.0.0.0"
