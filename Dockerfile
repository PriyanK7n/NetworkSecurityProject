FROM python:3.9.20-slim-bookworm

# Set the working directory inside the container to /app
WORKDIR /app

# Copy all files from the local directory to the container's /app directory
COPY . /app

# Update package list and install AWS CLI for interaction with AWS services
RUN apt update -y && apt install awscli -y

# Update package list and install Python dependencies from requirements.txt
RUN apt-get update && pip install -r requirements.txt

# Start the Python application (app.py), equivalent to running "uvicorn app:app" due to the app_run imported from uvicorn
CMD ["python3", "app.py"] 
