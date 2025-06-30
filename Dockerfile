FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDACXX=/usr/local/cuda/bin/nvcc
ENV VENV_PATH=/venv

RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    git \
    cmake \
    sox \
    ffmpeg \
    libsdl2-dev \
    libportaudio2 \
    portaudio19-dev \
    libopenblas-dev \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    zlib1g-dev \
    libffi-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    uuid-dev \
    liblzma-dev \
    python3-pyaudio \
    pulseaudio \
    ccache && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python 3.10.12 from source
WORKDIR /usr/src

RUN wget https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz && \
    tar xzf Python-3.10.12.tgz && \
    cd Python-3.10.12 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    rm -rf /usr/src/Python-3.10.12* /usr/src/Python-3.10.12.tgz

RUN python3.10 -m venv $VENV_PATH

RUN $VENV_PATH/bin/pip install --upgrade pip && \
    $VENV_PATH/bin/pip install --no-cache-dir \
        setuptools \
        wheel \
        fastapi \
        uvicorn \
        requests \
        "numpy<2" \
        scipy \
        python-dotenv \
        soundfile \
        cffi \
        webrtcvad \
        sounddevice \
        pydub \
        resampy \
        vosk==0.3.45 \
        pvporcupine==3.0.5 \
        tflite-runtime==2.14.0 \
        openwakeword==0.6.0

ENV PATH="$VENV_PATH/bin:$PATH"
WORKDIR /app

EXPOSE 5600

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5600"]
