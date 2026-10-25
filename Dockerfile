# Two-stage build: compile and test in a full toolchain image, ship only the
# CLI, the Python module and the example in a slim runtime image.
FROM ubuntu:24.04 AS build
RUN apt-get update -q && apt-get install -y -q --no-install-recommends \
      g++ cmake ninja-build git ca-certificates python3-dev python3-numpy \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cmake --preset release -DTICKFORGE_PYTHON=ON -DTICKFORGE_BUILD_BENCH=OFF \
    && cmake --build --preset release \
    && ctest --preset release

FROM ubuntu:24.04
LABEL org.opencontainers.image.source="https://github.com/henryzhangpku/tickforge" \
      org.opencontainers.image.description="tickforge: C++20 tick research engine (learning project, synthetic data)" \
      org.opencontainers.image.licenses="MIT"
RUN apt-get update -q && apt-get install -y -q --no-install-recommends python3 python3-numpy \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 tickforge
COPY --from=build /src/build/release/tickforge_cli /usr/local/bin/
COPY --from=build /src/build/release/tickforge*.so /opt/tickforge/
COPY python/examples /opt/tickforge/examples
ENV PYTHONPATH=/opt/tickforge
# Numeric UID so Kubernetes can enforce runAsNonRoot.
USER 10001
WORKDIR /home/tickforge
# Default: generate data, run one backtest and a sweep, then the Python example.
CMD ["sh", "-c", "tickforge_cli gen ticks.tf 1000000 42 && tickforge_cli backtest ticks.tf && tickforge_cli sweep ticks.tf | tail -1 && python3 /opt/tickforge/examples/research_example.py"]
