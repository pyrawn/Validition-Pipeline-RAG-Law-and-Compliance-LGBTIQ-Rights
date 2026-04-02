FROM apache/airflow:2.9.1-python3.11

COPY requirements.txt /requirements.txt

# Step 1 — System dependencies required by Docling for PDF processing.
USER root
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
USER airflow

# Step 2 — Install CPU-only torch and torchvision BEFORE resolving docling.
# Without this, docling's dependency resolver selects the CUDA variant of torch
# (~530 MB) plus nvidia_cudnn (~366 MB) and nvidia_cusparselt (~170 MB) —
# packages that are useless in a CPU-only Airflow container and reliably time
# out on slower connections.  Pre-installing the CPU wheel (~200 MB) causes pip
# to treat torch as already satisfied when it processes docling's requirements.
RUN pip install --no-cache-dir --timeout=300 \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Step 2 — Install everything else.  torch is already present so docling will
# not attempt to pull the CUDA build.
RUN pip install --no-cache-dir --timeout=300 -r /requirements.txt

# Copy full project into the container
COPY . /opt/airflow/project
