FROM python:3.11-slim

WORKDIR /app

# Tesseract + Bangla language pack, plus libs OpenCV needs at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-ben \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY streamlit_app.py .

EXPOSE 8000 8501

# Default: run the FastAPI service. Override CMD to run Streamlit instead:
#   docker run ... streamlit run streamlit_app.py --server.address 0.0.0.0
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
