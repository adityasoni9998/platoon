```bash
cd plugins/codescout
source .venv/bin/activate
export TINKER_API_KEY=tml-dummy
export TINKER_BASE_URL=http://localhost:9000
trial="skyrl-amd-codescout-lora-cispo-$(date +%Y%m%d-%H%M%S)"

uv run --extra tinker python -m platoon.codescout.train_tinker \
    --config platoon/codescout/train_codescout_tinker.yaml \
    --tinker_base_url $TINKER_BASE_URL \
    --stats.trial_name "$trial"
```

```bash
cd plugins/codescout
source .venv/bin/activate
export TINKER_API_KEY=tml-dummy
export TINKER_BASE_URL=http://localhost:9000

trial="skyrl-amd-codescout-lora-cispo-$(date +%Y%m%d-%H%M%S)"
export TINKER_PREFIX_MISMATCH_DEBUG_DIR="/data/user_data/adityabs/platoon_skyrl_tinker/logs/codescout-platoon-tinker/${trial}/prefix_mismatch_debug"
mkdir -p $TINKER_PREFIX_MISMATCH_DEBUG_DIR
TINKER_PREFIX_MISMATCH_DEBUG_DIR="/data/user_data/adityabs/platoon_skyrl_tinker/logs/codescout-platoon-tinker/${trial}/prefix_mismatch_debug" uv run --extra tinker python -m platoon.codescout.train_tinker \
    --config platoon/codescout/train_codescout_tinker.yaml \
    --tinker_base_url "$TINKER_BASE_URL" \
    --stats.trial_name "$trial"
```