#!/bin/bash
# Copyright AGNTCY Contributors (https://github.com/agntcy)
# SPDX-License-Identifier: Apache-2.0
set -ex

# Always run from the directory containing this script so relative paths work
# regardless of where the caller invokes it from.
cd "$(dirname "${BASH_SOURCE[0]}")"

export ITK_LOG_LEVEL="${ITK_LOG_LEVEL:-INFO}"
RESULT=1

cleanup() {
  set +x
  echo "Cleaning up artifacts..."
  docker stop itk-service > /dev/null 2>&1 || true
  docker rm itk-service > /dev/null 2>&1 || true
  # Preserve the image so layer caching speeds up subsequent local runs.
  # Set ITK_REMOVE_IMAGE=1 to force removal (e.g. in CI cleanup jobs).
  if [ "${ITK_REMOVE_IMAGE:-0}" = "1" ]; then
    docker rmi itk_service > /dev/null 2>&1 || true
  fi
  rm -rf a2a-itk > /dev/null 2>&1 || true
  echo "Done. Final exit code: $RESULT"
}
trap cleanup EXIT

: "${A2A_ITK_REVISION:?A2A_ITK_REVISION environment variable must be set}"

if [ ! -d "a2a-itk" ]; then
  git clone https://github.com/a2aproject/a2a-itk.git a2a-itk
fi
cd a2a-itk
git fetch origin
git checkout "$A2A_ITK_REVISION"
if git symbolic-ref -q HEAD > /dev/null; then
  git pull origin "$A2A_ITK_REVISION"
fi
cd ..

# Build the ITK service Docker image (polyglot: includes Rust 1.85, Go, Python, Node, etc.)
docker build -t itk_service a2a-itk

A2A_RS_ROOT=$(cd .. && pwd)
ITK_DIR=$(pwd)

docker rm -f itk-service || true

DOCKER_MOUNT_LOGS=""
if [ "${ITK_LOG_LEVEL^^}" = "DEBUG" ]; then
  mkdir -p "$ITK_DIR/logs"
  DOCKER_MOUNT_LOGS="-v $ITK_DIR/logs:/app/logs"
fi

docker run -d --name itk-service \
  -v "$A2A_RS_ROOT:/app/agents/repo" \
  -v "$ITK_DIR:/app/agents/repo/itk" \
  $DOCKER_MOUNT_LOGS \
  -e ITK_LOG_LEVEL="$ITK_LOG_LEVEL" \
  -p 8000:8000 \
  itk_service

docker exec -u root itk-service git config --system --add safe.directory /app/agents/repo
docker exec -u root itk-service git config --system --add safe.directory /app/agents/repo/itk
docker exec -u root itk-service git config --system core.multiPackIndex false

MAX_RETRIES=30
echo "Waiting for ITK service to start on 127.0.0.1:8000..."
set +e
for i in $(seq 1 $MAX_RETRIES); do
  if curl -s http://127.0.0.1:8000/ > /dev/null; then
    echo "Service is up!"
    break
  fi
  echo "Still waiting... ($i/$MAX_RETRIES)"
  sleep 2
done

if ! curl -s http://127.0.0.1:8000/ > /dev/null; then
  echo "Error: ITK service failed to start on port 8000"
  docker logs itk-service
  exit 1
fi

SCENARIO_FILE="scenarios.json"
if [ "${ITK_NIGHTLY_RUN^^}" = "TRUE" ]; then
  SCENARIO_FILE="scenarios_full.json"
fi

echo "ITK Service is up! Sending compatibility test request using $SCENARIO_FILE..."
RESPONSE=$(curl -s -X POST http://127.0.0.1:8000/run \
  -H "Content-Type: application/json" \
  -d "@$SCENARIO_FILE")

if [ "${ITK_NIGHTLY_RUN^^}" = "TRUE" ]; then
  echo "Nightly run detected. Saving raw results and processing history..."
  echo "$RESPONSE" > raw_results.json
  python3 process_results.py nightly \
    --scenarios  "$SCENARIO_FILE" \
    --output     itk_rust.json \
    --history-url https://github.com/a2aproject/a2a-rs/releases/download/nightly-metrics/itk_rust.json
  RESULT=$?
else
  echo "$RESPONSE" | python3 process_results.py ci
  RESULT=$?
fi
set -e

if [ $RESULT -ne 0 ]; then
  echo "Tests failed. Container logs:"
  docker logs itk-service
fi
echo "--------------------------------------------------------"

exit $RESULT
