## Installation

```bash
cd plugins/number-search
uv sync --extra tinker --extra wandb
source .venv/bin/activate
```

## Run Trainer

```bash
export TINKER_API_KEY=tml-dummy
export TINKER_BASE_URL=http://localhost:9000
trial="skyrl-amd-lora-cispo-$(date +%Y%m%d-%H%M%S)"
rm -rf ./logs/number-search-platoon-skyrl/$trial
mkdir -p ./logs/number-search-platoon-skyrl/$trial

uv run --active python -m platoon.number_search.train_tinker \
    --config platoon/number_search/number_search_tinker_cispo_lora.yaml \
    --tinker_base_url $TINKER_BASE_URL \
    --stats.trial_name "$trial"
```

For FFT, use [this](number_search_tinker_cispo_fft.yaml) config. It will give OOM errors on AMD 8xMI300 after 10-20 steps from vLLM engines with the default Tinker server command. But if you reduce the `gpu_memory_utilization` to 0.2 in the vLLM engine, then the code will run without crashing but will use ~70% of GPU memory on vLLM nodes. 