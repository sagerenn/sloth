# syntax=docker/dockerfile:1

# ─── build stage ──────────────────────────────────────────────────────────────
# BUILDPLATFORM lets buildx run the compile natively (fast) on the build host,
# then we cross to the target only when exporting via --platform. Binaries are
# architecture-specific, so for true multi-arch we relly on the multi-platform
# builder running the rust:1 image under QEMU for each TARGETARCH.
FROM rust:1-bookworm AS builder

ARG TARGETARCH
ARG TARGETVARIANT

# OpenSSL not needed (no TLS features), but keep pkg-config out of the way.
ENV CARGO_TERM_COLOR=always

WORKDIR /build

# Copy manifests first to cache dependency compilation across source changes.
COPY Cargo.toml Cargo.lock* ./
COPY src ./src
COPY examples ./examples
# Vendored path dependencies (the A2A Rust SDK). Required before `cargo build`.
COPY vendor ./vendor
COPY config.example.toml ./

# Build the release binary.
RUN cargo build --release \
 && cp target/release/sloth-agent /build/sloth-agent

# ─── runtime stage ────────────────────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates tini \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --system --uid 10001 --home /app sloth
WORKDIR /app
USER sloth

COPY --from=builder /build/sloth-agent /usr/local/bin/sloth-agent
COPY --from=builder /build/config.example.toml /app/config.example.toml

ENV SLOTH_LOG_FORMAT=text \
    SLOTH_LOG_FILTER=info,sloth_agent=info

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/sloth-agent"]
CMD ["--config", "/app/config.toml"]
