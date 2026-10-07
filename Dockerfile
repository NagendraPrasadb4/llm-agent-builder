FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Offline demo (no API keys needed)
CMD ["python", "examples/demo.py"]
