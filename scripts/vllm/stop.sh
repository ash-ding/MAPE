#!/usr/bin/env bash
# Stop the vLLM servers serve.sh started for <name>.
#   scripts/vllm/stop.sh <name>
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NAME="${1:?usage: stop.sh <served name>}"
RUN="$HERE/run/$NAME"
[ -f "$RUN/.pids" ] || { echo "no servers recorded for $NAME under $RUN" >&2; exit 1; }
while read -r pid; do
  [ -n "$pid" ] && kill -TERM "$pid" 2>/dev/null && echo "sent TERM to $pid" || true
done < "$RUN/.pids"
sleep 5
while read -r pid; do
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null && echo "killed $pid" || true
done < "$RUN/.pids"
rm -f "$RUN/.pids"
echo "stopped $NAME"
