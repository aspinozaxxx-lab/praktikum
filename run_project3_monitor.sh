#!/usr/bin/env bash
set -euo pipefail

cd /opt/prak

log_file="/opt/prak/project3_gpu_monitor.log"
rm -f "$log_file"

(
  while true; do
    date +%H:%M:%S
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader || true
    sleep "${GPU_MONITOR_INTERVAL:-1}"
  done
) > "$log_file" 2>&1 &
monitor_pid=$!

set +e
/opt/prak/.venv/bin/python /opt/prak/project3_train_server.py "$@"
status=$?
set -e

kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true

echo "MONITOR_TAIL"
tail -n 80 "$log_file"

exit "$status"
