# Dockerfile

# 1. Base Image
FROM python:3.10-slim

# 2. Set Working Directory
WORKDIR /app

# 3. Install System Dependencies (if needed - e.g., for specific library builds)
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# 4. Set environment variables (optional, can be set during runtime)
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 5. Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the entire application code
COPY ./app /app/app

# 7. Copy ML models (Alternatively, mount via volume)
COPY ./ml_models /app/ml_models

# 8. Expose the port the app runs on
EXPOSE 8000

# 9. Command to run the application using Uvicorn
#    Use --host 0.0.0.0 to make it accessible outside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]