# Use an official, lightweight Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's caching mechanism
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK corpora during build phase so they are baked into the image
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('vader_lexicon', quiet=True)"

# Copy the application source code and mock datasets
COPY app.py nlp_engine.py ./
COPY data/ ./data/
COPY static/ ./static/

# Expose port 8000 (standard port for Uvicorn/FastAPI)
EXPOSE 8000

# Start the application using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
