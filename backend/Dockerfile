FROM python:3.12

WORKDIR /app

COPY pyproject.toml .

COPY src ./src

COPY api ./api

RUN python -m pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

