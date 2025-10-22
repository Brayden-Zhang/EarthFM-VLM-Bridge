# The base image, which will be the starting point for the Docker image.
# We're using a PyTorch image built from https://github.com/allenai/docker-images
# because PyTorch is really big we want to install it first for caching.
FROM ghcr.io/allenai/pytorch:2.5.1-cuda12.1-python3.11

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# This is the directory that files will be copied into.
# It's also the directory that you'll start in if you connect to the image.
WORKDIR /stage/

COPY pyproject.toml /stage/pyproject.toml
COPY uv.lock /stage/uv.lock
RUN uv sync --all-groups --no-install-project

# Copy the folder `scripts` to `scripts/`
# You might need multiple of these statements to copy all the folders you need for your experiment.
COPY olmoearth_pretrain/ /stage/olmoearth_pretrain/
ENV PYTHONPATH="${PYTHONPATH}:/stage/"
RUN uv sync --all-groups --locked
