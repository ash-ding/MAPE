# Serving a local backbone with vLLM

A MAPE evaluation can run its unified LLM — the QA model and every LLM
call inside the memory system — on a locally served open model instead of
the OpenAI API. Serving is vLLM behind its OpenAI-compatible endpoint; the
evaluator routes by model name (`core/llm.py`), so nothing in a harness
changes: `Agent()` with `model` unset resolves to the run's unified model as
before, and the model policy still allows exactly that model. The judge
stays on its own (API) model, and embeddings stay on the OpenAI embedding
model — vLLM here serves chat only.

## Once per host

```bash
cd scripts/vllm && uv sync        # own venv; vLLM pinned to the line the host's CUDA driver can run
```

`pyproject.toml` pins `vllm==0.8.5.post1` / `torch==2.6.0` (CUDA 12.4): the
last line a 550-series NVIDIA driver (CUDA 12.4) runs, and the first with Qwen3.
Hosts with a newer driver can raise the pin.

## The backbone we serve

The project's second backbone is **Qwen3-30B-A3B-Instruct-2507**, served
under the name `qwen3-30b-a3b-instruct-2507`. It is a held-out-test setting only: an evolved
harness (evolved under gpt-4.1-mini) or a baseline in its unified arm is
re-run with the backbone swapped; nothing is evolved on it. bf16 weights are
~61 GB, so it needs two 48 GB GPUs (`--tp 2`) or one 80 GB GPU:

```bash
uvx --from huggingface_hub hf download Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --local-dir /path/to/models/Qwen3-30B-A3B-Instruct-2507
scripts/vllm/serve.sh --model /path/to/models/Qwen3-30B-A3B-Instruct-2507 \
    --name qwen3-30b-a3b-instruct-2507 --gpus 0,1 --tp 2 --port-base 8300
```

The Instruct-2507 line has no thinking mode, so `serve.sh` passes no
`chat_template_kwargs` for it (hybrid Qwen3 models such as Qwen3-8B get
`enable_thinking: false`). Its price on the cost axis is the list price in
`core/pricing.py`, not a route price.

## Serve (any model)

```bash
# weights: a local dir or a Hugging Face id (downloaded into --hf-home)
scripts/vllm/serve.sh --model /path/to/models/Qwen3-8B --name qwen3-8b \
    --gpus 0,1 --port-base 8300 --max-model-len 32768 --hf-home /path/to/hf-cache
```

One server per listed GPU (or `--tp N` for one server over N GPUs). It waits
for every server to answer `/v1/models`, then writes
`scripts/vllm/run/<name>/routes.json` and prints the `llm.routes` block to put
in the run config:

```yaml
model: qwen3-8b                 # the unified LLM: QA model + memory system
judge_model: gpt-5-mini         # unchanged
llm:
  routes:
    qwen3-8b:
      base_urls: ["http://127.0.0.1:8300/v1", "http://127.0.0.1:8301/v1"]
      api_key: EMPTY
      chat_template_kwargs: {"enable_thinking": false}   # Qwen3: no <think> preamble
      price_usd_per_1m: [0.10, 0.30]                      # what the cost axis charges it (see below)
```

The evaluator serialises `llm.routes` into `MAPE_MODEL_ROUTES` for every
container; `core.llm` round-robins calls over `base_urls`. Containers share
the host network (apptainer default), so `127.0.0.1` reaches the servers.

A **baseline** takes the same table as the optional `routes:` key of its
frame config (`baselines/config.example.yaml`), with
`unified_models.llm` and `llm_model` set to the served name. Its vendored
OpenAI clients are redirected at the SDK boundary by `core.openai_usage`
(chat completions only — zep's graphiti uses the Responses API, which vLLM
does not implement, so zep cannot run on the local backbone; the shim
refuses with a clear error).

`scripts/vllm/stop.sh qwen3-8b` stops them.

## Cost of a local model

The cost axis prices tokens (`core/pricing.py`); a served model has no
list price. Give it one in its route (`price_usd_per_1m: [input, output]`)
— an API-equivalent rate is the honest choice, so a local backbone is not
"free" on the frontier. Without it the model is charged at the unified
LLM's rate if that is priced, else 0, and is named in
`cost_unpriced_models`.

## Notes

- Qwen3 chat templates default to thinking; `serve.sh` turns it off for
  names starting `qwen3` (`--thinking` to keep it). A thinking preamble would
  otherwise be counted as output tokens and break "short answer" prompts.
- `max_completion_tokens`, `response_format: json_object` and
  `reasoning_effort` (ignored for non-reasoning names) go through vLLM's
  OpenAI server as they do to OpenAI.
- Logs: `scripts/vllm/run/<name>/server-<port>.log`.
