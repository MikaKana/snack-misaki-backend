# syntax=docker/dockerfile:1.6

FROM ubuntu:22.04 AS python-base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        curl \
        git \
        build-essential \
        cmake \
        pkg-config \
        gnupg \
        libcurl4-openssl-dev \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        python3.11-distutils \
        python3-pip \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && curl -Lo /usr/local/bin/aws-lambda-rie "https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/latest/download/aws-lambda-rie" \
    && chmod +x /usr/local/bin/aws-lambda-rie \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

FROM python-base AS base

COPY pyproject.toml README.md LICENSE ./
COPY app ./app

FROM base AS build-runtime

RUN python3 -m pip install --no-cache-dir --target /opt/python awslambdaric \
    && python3 -m pip install --no-cache-dir --target /opt/python .

FROM base AS build-development
RUN python3 -m pip install --no-cache-dir --target /opt/python awslambdaric \
    && python3 -m pip install --no-cache-dir --target /opt/python ".[dev]"

FROM python-base AS llama-build

RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
WORKDIR /app/llama.cpp
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --target llama-cli

FROM python-base AS runtime

COPY --from=build-runtime /opt/python /opt/python
COPY --from=llama-build /app/llama.cpp /app/llama.cpp
COPY app /app/app
COPY README.md /app/README.md
COPY LICENSE /app/LICENSE

ENV PYTHONPATH="/opt/python:/app:${PYTHONPATH}"
ENV PATH="/app/llama.cpp/build/bin:${PATH}"

CMD ["python3", "-m", "awslambdaric", "app.handler.lambda_handler"]

FROM python-base AS development

COPY --from=build-development /opt/python /opt/python
COPY --from=llama-build /app/llama.cpp /app/llama.cpp
COPY app /app/app
COPY tests /app/tests
COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md
COPY LICENSE /app/LICENSE

ENV PYTHONPATH="/opt/python:/app:${PYTHONPATH}"
ENV PATH="/app/llama.cpp/build/bin:${PATH}"

CMD ["/usr/local/bin/aws-lambda-rie", "python3", "-m", "awslambdaric", "app.handler.lambda_handler"]