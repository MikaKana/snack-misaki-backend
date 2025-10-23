# syntax=docker/dockerfile:1.6

FROM public.ecr.aws/lambda/python:3.11 AS base

WORKDIR /var/task

COPY pyproject.toml README.md LICENSE ./
COPY app ./app

RUN python -m pip install --upgrade pip

FROM base AS build-runtime
RUN python -m pip install --no-cache-dir --target /opt/python .

FROM base AS build-development
RUN python -m pip install --no-cache-dir --target /opt/python ".[dev]"

FROM public.ecr.aws/lambda/python:3.11 AS runtime
COPY --from=build-runtime /opt/python /opt/python
COPY app /var/task/app
COPY README.md /var/task/README.md
COPY LICENSE /var/task/LICENSE
ENV PYTHONPATH="/opt/python:${PYTHONPATH}"
CMD ["app.handler.lambda_handler"]

FROM public.ecr.aws/lambda/python:3.11 AS development
COPY --from=build-development /opt/python /opt/python
COPY app /var/task/app
COPY README.md /var/task/README.md
COPY LICENSE /var/task/LICENSE
ENV PYTHONPATH="/opt/python:${PYTHONPATH}"
CMD ["app.handler.lambda_handler"]