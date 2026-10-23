FROM python:3.12-slim
RUN python -m pip install --no-cache-dir pytest==8.3.5
ENV PYTHONDONTWRITEBYTECODE=1
