FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
ARG INSTALL_MINERU=false
COPY requirements-document-ai.txt .
RUN if [ "$INSTALL_MINERU" = "true" ]; then pip install --no-cache-dir -r requirements-document-ai.txt; fi
COPY app ./app
COPY data ./data
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
