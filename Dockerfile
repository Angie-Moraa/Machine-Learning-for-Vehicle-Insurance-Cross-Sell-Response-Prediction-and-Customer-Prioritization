FROM python:3.11-slim

# LightGBM system library
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# installed dependencies first (cached between builds)
COPY requirements.txt setup.py ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt

# the app, the form and the trained model
COPY app.py ./
COPY templates ./templates
COPY final_model ./final_model

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]