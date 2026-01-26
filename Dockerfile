# Ubuntu 22.04 alap
FROM ubuntu:22.04

# Környezeti változók
ENV DEBIAN_FRONTEND=noninteractive
ENV SUMO_HOME=/usr/share/sumo
# TÖRÖLVE: ENV PYTHONPATH=$SUMO_HOME/tools:$PYTHONPATH
# (Így a pip-es libsumo fog betöltődni, ami működik)

# 1. Rendszerfüggőségek
RUN apt-get update && apt-get install -y \
    software-properties-common \
    python3 \
    python3-pip \
    git \
    wget \
    ffmpeg \
    && add-apt-repository ppa:sumo/stable -y \
    && apt-get update \
    && apt-get install -y sumo sumo-tools sumo-doc \
    && rm -rf /var/lib/apt/lists/*

# Szimbolikus link python -> python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# 2. Munkakönyvtár
WORKDIR /app

# 3. Python csomagok
COPY requirements.txt .
# Hozzáadjuk a 'sumolib'-et is, ha a requirements-ben nem lenne benne
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir libsumo sumolib

# 4. Forráskód másolása
COPY training_config.json .
COPY . .


# 5. Indítás
CMD ["python", "main_headless.py", "--config", "training_config.json"]