#!/bin/bash
set -euo pipefail
set -x

export PYTHONUNBUFFERED=1
export EXCHANGE_HOST=${EXCHANGE_HOST:-127.0.0.1}
export EXCHANGE_PORT=${EXCHANGE_PORT:-18080}
export EXCHANGE_MAX_QUEUE_SIZE=${EXCHANGE_MAX_QUEUE_SIZE:-20000}

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ] && [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
fi
if [ -z "$PYTHON_BIN" ] && [ -x "/zhangshihao/weitong/anaconda3/envs/verl/bin/python" ]; then
  PYTHON_BIN="/zhangshihao/weitong/anaconda3/envs/verl/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"

$PYTHON_BIN -m verl.experimental.fully_async_policy.tcp_exchange_server_main \
  --host "$EXCHANGE_HOST" \
  --port "$EXCHANGE_PORT"

