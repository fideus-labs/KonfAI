FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# What `konfai[...]` pulls: the imaging backends (SimpleITK, h5py, pydicom, zarr, ngff-zarr, dask).
ARG KONFAI_EXTRAS=imaging
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

WORKDIR /tmp

RUN apt-get update \
    && apt-get install -y --no-install-recommends tini libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/entrypoint.sh /usr/local/bin/konfai-entrypoint

# The wheels this image ships, taken from the build context rather than from PyPI. The version is
# then whatever was built -- nothing to keep in sync with the tag, and no waiting for an index to
# serve what was uploaded a moment ago. `python -m build` puts them here; see docker/README.md.
# Only the two installed below: a layer keeps every file it copies, whatever a later step deletes.
COPY dist/konfai-*.whl dist/konfai_apps-*.whl /tmp/dist/

RUN chmod +x /usr/local/bin/konfai-entrypoint \
    # Resolved through a variable because the shell would read the `[...]` of a glob as a character
    # class. `konfai-*` cannot match a sibling: a wheel spells `konfai-apps` as `konfai_apps`. Each
    # local build leaves its own version in dist/, so the count is checked before torch is installed.
    && wheel_of() { \
         set -- /tmp/dist/"$1"-*.whl; \
         [ "$#" -eq 1 ] && [ -e "$1" ] && echo "$1" && return; \
         echo "dist/ must hold one wheel of each package, it holds:" $(ls /tmp/dist) >&2; \
         echo "Empty dist/ and build the wheels again (docker/README.md)." >&2; \
         return 1; \
       } \
    && konfai_wheel="$(wheel_of konfai)" \
    && apps_wheel="$(wheel_of konfai_apps)" \
    && pip install --upgrade pip setuptools wheel \
    && pip install --index-url "${TORCH_INDEX_URL}" --extra-index-url https://pypi.org/simple torch \
    && pip install "${konfai_wheel}[${KONFAI_EXTRAS}]" "${apps_wheel}[server]" \
    && rm -rf /tmp/dist

WORKDIR /workspace

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/konfai-entrypoint"]
CMD ["konfai", "--help"]
