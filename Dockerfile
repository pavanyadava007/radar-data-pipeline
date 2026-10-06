# CPU image: runs the lint + test suite (synthetic fixture, no download) and exposes the CLI.
FROM python:3.11-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
WORKDIR /app
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt
COPY . .
RUN pip install --no-deps -e .
CMD ["sh", "-c", "ruff check . && pytest -q && rdp --help"]
