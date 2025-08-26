FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data cache logs

# Expose ports
EXPOSE 8000 8001

# Run the application
CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]