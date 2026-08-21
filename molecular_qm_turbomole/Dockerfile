# Build from this capability repository:
#   docker build -t molecular-qm-turbomole:latest .
# From simstack-model:
#   docker build -t molecular-qm-turbomole:latest -f molecular_qm_turbomole/Dockerfile molecular_qm_turbomole
FROM mambaorg/micromamba:latest

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN micromamba install -y -n base -c conda-forge setuptools && \
    micromamba clean --all --yes

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/opt/conda/bin:/root/.local/bin:$PATH"

WORKDIR /app

RUN micromamba install -y -n base -c conda-forge python=3.12 && \
    micromamba clean --all --yes

ENV UV_PYTHON=/opt/conda/bin/python
ENV UV_PROJECT_ENVIRONMENT=/opt/conda
ENV PATH="/opt/conda/bin:/root/.local/bin:$PATH"

COPY . /build/molecular_qm_turbomole
WORKDIR /build/molecular_qm_turbomole
RUN cp pyproject.docker pyproject.toml \
 && uv pip install --system . "setuptools>=80.9.0" \
 && python -c "import simstack, molecular_qm_models, molecular_qm_turbomole"

WORKDIR /app
ENTRYPOINT ["python", "-m", "simstack.core.run_node"]
