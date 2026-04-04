#!/usr/bin/env bash
set -euo pipefail

_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${_REPO_ROOT}/verl${PYTHONPATH:+:${PYTHONPATH}}"

export PYTHONUNBUFFERED=1
EXCHANGE_HOST="${EXCHANGE_HOST:-0.0.0.0}"
EXCHANGE_PORT="${EXCHANGE_PORT:-28080}"
_raw_max="${EXCHANGE_MAX_QUEUE_SIZE:-20000}"
# 只保留数字，防止异常 env / IFS 拆成「2」和「0000」两条参数
EXCHANGE_MAX_QUEUE_SIZE="${_raw_max//[^0-9]/}"
[[ -z "${EXCHANGE_MAX_QUEUE_SIZE}" || "${EXCHANGE_MAX_QUEUE_SIZE}" -lt 100 ]] && EXCHANGE_MAX_QUEUE_SIZE=20000
export EXCHANGE_HOST EXCHANGE_PORT EXCHANGE_MAX_QUEUE_SIZE

# 无 sudo：尽量释放本用户对 ${EXCHANGE_PORT} 的监听（多等一下，避免 TIME_WAIT / 内核未完全回收）
_free_port() {
  local p="${1:?port}"
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "${p}/tcp" 2>/dev/null || true
  fi
  if command -v lsof >/dev/null 2>&1; then
    while read -r _pid; do
      [[ -n "${_pid}" ]] && kill "${_pid}" 2>/dev/null || true
    done < <(lsof -t -iTCP:"${p}" -sTCP:LISTEN 2>/dev/null || true)
  fi
  sleep 0.5
}

_free_port "${EXCHANGE_PORT}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" && -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
fi
if [[ -z "${PYTHON_BIN}" && -x "/zhangshihao/weitong/anaconda3/envs/verl/bin/python" ]]; then
  PYTHON_BIN="/zhangshihao/weitong/anaconda3/envs/verl/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[run_exchange_server] bind=${EXCHANGE_HOST} port=${EXCHANGE_PORT} max_queue_size=${EXCHANGE_MAX_QUEUE_SIZE}" >&2
echo "[run_exchange_server] 训练端 A/B 请设置能连到本进程的 host/port；多 Pod 勿用各 Pod 内 127.0.0.1。" >&2

# 单行 exec，避免续行被拆成多条命令；占住终端直到服务端退出
exec "${PYTHON_BIN}" -m verl.experimental.fully_async_policy.tcp_exchange_server_main --host "${EXCHANGE_HOST}" --port "${EXCHANGE_PORT}" --max-queue-size "${EXCHANGE_MAX_QUEUE_SIZE}"
