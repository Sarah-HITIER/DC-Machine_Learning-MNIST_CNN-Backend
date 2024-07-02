# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA 12.1
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy the entire backend directory into the container
COPY . .

# Expose the port that FastAPI runs on
EXPOSE 8000

# Command to run FastAPI app
# CMD ["fastapi", "dev", "src/api.py"]
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]