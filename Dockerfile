FROM ubuntu:22.04

# Set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# Install Python first
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Update package sources with multiple mirrors
RUN sed -i 's|archive.ubuntu.com|mirrors.ubuntu.com/mirrors.txt|g' /etc/apt/sources.list || true && \
    sed -i 's|security.ubuntu.com|mirrors.ubuntu.com/mirrors.txt|g' /etc/apt/sources.list || true

# Install system dependencies with retry logic
RUN for i in 1 2 3; do \
    apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        && break || sleep 10; \
    done && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install Python packages
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir "paddleocr>=2.8.0" && \
    pip3 install --no-cache-dir torch==2.1.0+cpu torchvision==0.16.0+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    pip3 install --no-cache-dir -r requirements.txt

# Create symbolic link for python
RUN ln -sf /usr/bin/python3 /usr/bin/python

COPY *.py .

RUN mkdir -p /app/cache /app/uploads

ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4

EXPOSE 7860

CMD ["python", "app.py"]