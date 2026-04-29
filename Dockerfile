# syntax=docker/dockerfile:1.6
# Memoirs — local-first memory engine.
# Image ships with: Python 3.12, sqlite-vec, sentence-transformers, spaCy en+es models.
# Gemma GGUF is NOT included (1.6 GB). Mount or download separately.
#
# Build:   docker build -t memoirs .
# Run:     docker run -it --rm -v "$HOME/.local/share/memoirs:/root/.local/share/memoirs" \
#                     -v "$(pwd)/.memoirs:/data" memoirs
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MEMOIRS_DB=/data/memoirs.sqlite

# System deps for compilation + Vulkan (CPU-only build by default — GPU passthrough
# requires --device + matching host driver, which is platform-specific).
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      git \
      ca-certificates \
      libvulkan-dev \
      glslc \
      glslang-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/memoirs

# Install Python deps in one layer for cache efficiency
COPY pyproject.toml ./
RUN pip install --upgrade pip \
    && pip install -e ".[embeddings,extract,viz]"

# spaCy models cached in image (saves ~50 MB download per container)
RUN python -m spacy download en_core_web_sm \
    && python -m spacy download es_core_news_sm

# Now copy code (changes here don't invalidate the deps layers above)
COPY memoirs/ ./memoirs/
RUN pip install -e .

# Default DB location: /data (mount a volume here)
RUN mkdir -p /data
VOLUME ["/data", "/root/.local/share/memoirs"]

# Default command: run MCP server on stdio
ENTRYPOINT ["memoirs", "--db", "/data/memoirs.sqlite"]
CMD ["mcp"]
