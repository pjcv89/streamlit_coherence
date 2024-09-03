# streamlit_coherence/Dockerfile

FROM python:3.10-slim
MAINTAINER Pablo Campos Viana

WORKDIR /streamlit_coherence

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/pjcv89/streamlit_coherence .

RUN pip3 install -r requirements.txt

RUN gdown 1UwI02AW08g0EAh6DYIkHtgzQtwWM_Pz6 && \
    unzip artifacts.zip && \
    rm artifacts.zip

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "Welcome.py", "--server.port=8501", "--server.address=0.0.0.0"]