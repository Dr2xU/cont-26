FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Generate available 2D exports, then run trustworthiness comparison.
CMD ["sh", "-c", "python generate.py && python evaluate.py"]
