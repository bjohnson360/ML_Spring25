# Use official PyTorch image with CUDA support
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# For CPU version
#FROM python:3.10-slim

# For CPU version
#RUN apt-get update && apt-get install -y \
#    libglib2.0-0 libsm6 libxrender1 libxext6 \
#    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only code and config files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# For CPU version
#RUN pip install --no-cache-dir \
#    torch==2.1.0+cpu torchvision==0.16.0+cpu -f https://download.pytorch.org/whl/torch_stable.html \
#    facenet-pytorch pillow matplotlib

# Default command to run face matcher
#CMD ["python", "app.py", "--image", "data/sample.jpg"]
CMD ["python", "extract_fddb_faces.py"]