# Use a minimal base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

COPY requirements.txt /app
COPY ./ /app/
# Install the necessary dependencies
RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install numpy
# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define the command to run the app
CMD ["uvicorn", "model:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info", "--reload"]

