# Use a minimal base image
FROM python:3.11-slim

RUN mkdir app
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
# Define the command to run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info", "--reload"]

