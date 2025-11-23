FROM python:3.12

RUN python -m pip install --upgrade pip

COPY . /app
WORKDIR /app
RUN python -m pip install .[ui]
RUN python -m gilda.resources && \
  python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"
ENTRYPOINT ["python", "-m", "gilda.app", "--port", "8001", "--host", "0.0.0.0"
