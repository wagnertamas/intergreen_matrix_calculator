# Ubuntu 22.04 alap
FROM ubuntu:22.04

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

# 3. Python csomagok
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir libsumo sumolib pyyaml

# 4. Forráskód másolása
COPY . .

# 5. Indítás — a config-ot a docker-compose command felülírhatja
CMD ["python", "main_headless.py", "--config", "training_config.yaml"]
