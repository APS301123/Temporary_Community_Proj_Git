FROM python:3.11-slim

WORKDIR /app

# OpenCV system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default data dir — override with a volume in production
RUN mkdir -p /data/uploads

ENV UPLOAD_DIR=/data/uploads
ENV DATABASE_PATH=/data/lostnfound.db

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
