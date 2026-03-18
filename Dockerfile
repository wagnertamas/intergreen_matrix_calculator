# =============================================================================
# SUMO RL Training — CUDA-aware Dockerfile
# =============================================================================
# Két variáns:
#   1. GPU:  docker-compose --profile gpu up   (NVIDIA GPU + CUDA)
#   2. CPU:  docker-compose up                  (alap, mindig működik)
#
# A kód futásidőben detektálja a GPU-t:
#   torch.cuda.is_available() → True → CUDA device
#   torch.backends.mps.is_available() → True → Apple MPS (nem Docker-ben)
#   egyébként → CPU
# =============================================================================

# --- Base image: CUDA ha elérhető, egyébként Ubuntu ---
ARG BASE_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV SUMO_HOME=/usr/share/sumo

# 1. Rendszerfüggőségek + SUMO
RUN apt-get update && apt-get install -y \
    software-properties-common \
    python3 \
    python3-pip \
    git \
    wget \
    && add-apt-repository ppa:sumo/stable -y \
    && apt-get update \
    && apt-get install -y sumo sumo-tools sumo-doc \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# 2. Munkakönyvtár
WORKDIR /app

# 3. Python csomagok — PyTorch CUDA-val
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir libsumo libtraci sumolib pyyaml

# 4. Forráskód másolása
COPY . .

# 5. GPU detektálás startup script
RUN echo '#!/bin/bash\n\
python3 -c "\n\
import torch\n\
if torch.cuda.is_available():\n\
    print(f\"[GPU] CUDA {torch.version.cuda} | {torch.cuda.get_device_name(0)}\")\n\
    print(f\"[GPU] Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB\")\n\
else:\n\
    print(\"[CPU] No CUDA GPU detected — using CPU\")\n\
"\n\
exec "$@"' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]

# 6. Default parancs — docker-compose command felülírhatja
CMD ["echo", "Használd: ./start.sh vagy docker-compose run --rm sumo-trainer <parancs>"]
