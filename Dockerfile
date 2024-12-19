FROM python:3.12-slim

WORKDIR /german_credit_scoring

RUN apt-get update -qq && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# RUN git clone https://github.com/KostyaGukish/German-Credit-Scoring.git .
COPY . .

RUN pip3 install -r requirements.txt
# RUN pip3 install -r requirements.txt
# conda create --name <env> --file <this file>

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit/application.py", "--server.port=8501", "--server.address=0.0.0.0"]