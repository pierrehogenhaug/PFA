# Use a slim Python base image
FROM python:3.9-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy only the requirements first (to leverage Docker cache)
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application code
COPY . /app

# Expose port 8000 so that container orchestration systems know which port to map
EXPOSE 8000

# Run the application using Uvicorn with the exec form.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]