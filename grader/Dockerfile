# Use an official Python runtime as the base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH "${PYTHONPATH}:/grader"

# Set the working directory inside the container
WORKDIR /grader

# Copy the requirements file into the container
COPY requirements.txt .

# Install FastAPI and Uvicorn (and any other dependencies)
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the current directory (i.e., source folder) contents into the container
COPY . /grader

# Specify the command to run on container start
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
