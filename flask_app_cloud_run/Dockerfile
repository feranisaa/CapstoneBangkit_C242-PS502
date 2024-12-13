# Use a Python base image
FROM python:3.11-slim


# Install system dependencies required for building and installing packages
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    python3-distutils \
    libatlas-base-dev \
    libopenblas-dev \
    libomp-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*


# Upgrade pip and setuptools
RUN pip install --upgrade pip
RUN pip install setuptools==67.8.0


# Set the working directory in the container
WORKDIR /flask_app_cloud_run

# Copy the requirements.txt into the container
COPY requirements.txt .

# Install dependencies directly into the global Python environment
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user and group
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Copy the application code into the container
COPY . .

# Set permissions on the application directory for the new user
RUN chown -R appuser:appgroup /flask_app_cloud_run

# Switch to non-root user for running the app
USER appuser

# Set environment variables (optional)
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Expose the port that the app will run on
EXPOSE 8080

# Command to run the Flask app
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]
