FROM python:3.12

WORKDIR /code

COPY pyproject.toml .

COPY src ./src

COPY backend ./backend

RUN python -m pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

