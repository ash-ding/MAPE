#!/usr/bin/env bash
# Serve a local backbone for MAPE with vLLM, one OpenAI-compatible server
# per GPU (or one tensor-parallel server over several), and write the routes
# file the evaluator hands to every container.
#
#   scripts/vllm/serve.sh --model /path/or/hf-id --name qwen3-8b [--gpus 0,1,2,3]
#                       [--tp 1] [--port-base 8300] [--max-model-len 32768]
#                       [--gpu-mem 0.90] [--hf-home DIR] [--no-thinking]
#
#   --gpus     GPUs to use. With --tp 1 (default) each GPU gets its own server
#              on port-base+i and the routes file lists them all; the client
#              round-robins across them. With --tp N one server spans N GPUs.
#   --name     the model name harnesses will ask for (cfg.model / the unified
#              LLM). Keep it short and stable; it is what the policy allows.
#   --no-thinking  for Qwen3-style models: pass chat_template_kwargs
#              {"enable_thinking": false} on every request so answers do not
#              start with a <think> block. Default ON for names starting
#              "qwen3"; --thinking turns it back on.
#
# Writes:
#   scripts/vllm/run/<name>/routes.json   → MAPE_MODEL_ROUTES content, and
#                                         the `llm.routes` block for a config
#   scripts/vllm/run/<name>/server-<port>.log, .pids
#
# Environment: `uv sync` in scripts/vllm once (its own venv, pinned to the
# vLLM line the hosts' CUDA driver can run — see pyproject.toml).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL=""; NAME=""; GPUS=""; TP=1; PORT_BASE=8300; MAX_LEN=32768; GPU_MEM=0.90
HF_HOME_DIR="${HF_HOME:-}"; THINKING="auto"; EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --name) NAME="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --tp) TP="$2"; shift 2;;
    --port-base) PORT_BASE="$2"; shift 2;;
    --max-model-len) MAX_LEN="$2"; shift 2;;
    --gpu-mem) GPU_MEM="$2"; shift 2;;
    --hf-home) HF_HOME_DIR="$2"; shift 2;;
    --no-thinking) THINKING="off"; shift;;
    --thinking) THINKING="on"; shift;;
    --) shift; EXTRA=("$@"); break;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -n "$MODEL" ] && [ -n "$NAME" ] || { echo "usage: $0 --model <path|hf-id> --name <served name> [--gpus 0,1] ..." >&2; exit 2; }
if [ -z "$GPUS" ]; then GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)"; fi
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
# Qwen3 hybrid models (Qwen3-8B, -32B, ...) think by default and need
# enable_thinking=false; the *-Instruct-2507 line has no thinking mode and
# its template takes no such switch, so nothing is passed for it.
if [ "$THINKING" = "auto" ]; then
  case "$NAME" in
    *instruct-2507*|*Instruct-2507*) THINKING="none";;
    qwen3*|Qwen3*) THINKING="off";;
    *) THINKING="on";;
  esac
fi

RUN="$HERE/run/$NAME"; mkdir -p "$RUN"
[ -n "$HF_HOME_DIR" ] && export HF_HOME="$HF_HOME_DIR"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

# One server per GPU (tp=1) or one server over the first TP GPUs.
declare -a PORTS=()
if [ "$TP" -gt 1 ]; then
  GROUPS_=("$(IFS=,; echo "${GPU_LIST[*]:0:$TP}")")
else
  GROUPS_=("${GPU_LIST[@]}")
fi
: > "$RUN/.pids"
i=0
for g in "${GROUPS_[@]}"; do
  port=$((PORT_BASE + i))
  log="$RUN/server-$port.log"
  echo "starting $NAME on GPU(s) $g → :$port  (log: $log)"
  CUDA_VISIBLE_DEVICES="$g" nohup uv run --project "$HERE" vllm serve "$MODEL" \
      --served-model-name "$NAME" --host 127.0.0.1 --port "$port" \
      --tensor-parallel-size "$TP" --max-model-len "$MAX_LEN" \
      --gpu-memory-utilization "$GPU_MEM" --dtype bfloat16 \
      --disable-log-requests "${EXTRA[@]}" > "$log" 2>&1 < /dev/null &
  echo "$!" >> "$RUN/.pids"
  PORTS+=("$port"); i=$((i + 1))
done

# Wait until every server answers /v1/models (weights load in the background).
echo "waiting for ${#PORTS[@]} server(s) to load ..."
for port in "${PORTS[@]}"; do
  for _ in $(seq 1 240); do
    if curl -s -m 3 "http://127.0.0.1:$port/v1/models" 2>/dev/null | grep -q "\"$NAME\""; then break; fi
    sleep 5
  done
  curl -s -m 3 "http://127.0.0.1:$port/v1/models" 2>/dev/null | grep -q "\"$NAME\"" \
    || { echo "server on :$port did not come up; see $RUN/server-$port.log" >&2; exit 1; }
  echo "  :$port ready"
done

# The routes file: what the evaluator puts in MAPE_MODEL_ROUTES.
python3 - "$RUN/routes.json" "$NAME" "$THINKING" "${PORTS[@]}" <<'PY'
import json, sys
out, name, thinking, *ports = sys.argv[1:]
route = {"base_urls": [f"http://127.0.0.1:{p}/v1" for p in ports], "api_key": "EMPTY"}
if thinking == "off":
    route["chat_template_kwargs"] = {"enable_thinking": False}
json.dump({name: route}, open(out, "w"), indent=2)
print(f"routes → {out}")
print("config snippet:")
print("llm:")
print("  routes:")
print(f"    {name}:")
print(f"      base_urls: {json.dumps(route['base_urls'])}")
print(f"      api_key: EMPTY")
if "chat_template_kwargs" in route:
    print(f"      chat_template_kwargs: {json.dumps(route['chat_template_kwargs'])}")
PY
echo "stop with: $HERE/stop.sh $NAME"
