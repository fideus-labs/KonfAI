FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

ARG KONFAI_EXTRAS=imaging
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG KONFAI_PYPI_VERSION=1.6.0

WORKDIR /tmp

RUN apt-get update \
    && apt-get install -y --no-install-recommends tini libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/entrypoint.sh /usr/local/bin/konfai-entrypoint

RUN chmod +x /usr/local/bin/konfai-entrypoint \
    && pip install --upgrade pip setuptools wheel \
    && pip install --index-url "${TORCH_INDEX_URL}" --extra-index-url https://pypi.org/simple torch \
    && pip install "konfai[${KONFAI_EXTRAS}]==${KONFAI_PYPI_VERSION}" "konfai-apps==${KONFAI_PYPI_VERSION}"

WORKDIR /workspace

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/konfai-entrypoint"]
CMD ["konfai", "--help"]
