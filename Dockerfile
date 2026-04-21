# Use a slim, stable Python 3.12 image for 2026
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies if needed (SQLite is built-in)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port Flask/Gunicorn will run on
EXPOSE 5000

# Use Gunicorn for production-grade serving
# -w 4: 4 worker processes
# -b 0.0.0.0:5000: Bind to all interfaces on port 5000
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:5000", "app:app"]