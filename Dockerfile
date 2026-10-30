FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Set environment caches inside /app/.cache
ENV MPLCONFIGDIR=/app/.cache/matplotlib
ENV KERAS_HOME=/app/.cache/keras
RUN mkdir -p /app/.cache/matplotlib /app/.cache/keras && chmod -R 777 /app/.cache

# Hugging Face default port is 7860
EXPOSE 7860

# Launch Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
