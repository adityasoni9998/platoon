# Legacy issue resolution on Sail

Same 257 easy tasks, default OpenHands prompt/tools, binary reward, training
settings, and subprocess rollouts as legacy. Agent workspaces use Sail;
test execution uses the existing Modal `swesmith-evaluation` deployment.
The automatic Tinker inference tunnel still uses Modal.

```bash
cd plugins/issue-resolution-legacy-sail
export UV_PROJECT_ENVIRONMENT=/tmp/issue-resolution-legacy-sail-tinker-venv
uv sync --locked --extra tinker
export MODAL_PROFILE=neulab
export OPENHANDS_SUPPRESS_BANNER=1
export MODAL_SANDBOX_V2=1
export ARROW_DEFAULT_MEMORY_POOL=system
export SGLANG_FORWARD_UNKNOWN_TOOLS=true
```

Use your default `~/.sail` credentials and configure Modal, `TINKER_API_KEY`,
and `WANDB_API_KEY`. Set `SGLANG_FORWARD_UNKNOWN_TOOLS=true` on the inference
server too if it uses SGLang; a client environment variable cannot configure
an already-running remote server.

Build/push once (only deduplicated images used by the easy subset; no named
Modal images). Requires Modal secret `dockerhub-adityasoni8` containing
`REGISTRY_USERNAME` and `REGISTRY_PASSWORD`, with push access to the public
DockerHub repository `adityasoni8/eval-agent-server`:

```bash
uv run --locked --extra tinker modal run scripts/build_agent_images.py --max-workers 8
```

Then train, after exporting the endpoint supplied separately:

```bash
: "${TINKER_BASE_URL:?Export your Tinker service URL before running}"
uv run --locked --extra tinker python -m platoon.issue_resolution_sail.train_tinker \
  --tinker_base_url "$TINKER_BASE_URL" \
  --stats.trial_name "legacy-sail-tinker-$(date +%Y%m%d-%H%M%S)"
```

SDK and image tags use `d3fd1c497860fdcba49e3304f348b10a4f1c370b`.
Sail defaults: size `s`, 2 GiB memory, 32 GiB disk; override with
`SAILBOX_SIZE`, `SAILBOX_MEMORY_GIB`, `SAILBOX_DISK_GIB`.
Image import: 1,800 seconds; creation/startup: 600 seconds each
(`SAIL_IMAGE_BUILD_TIMEOUT`, `SAIL_CREATE_TIMEOUT`, `SAIL_STARTUP_TIMEOUT`).
Legacy setup retries, episode/step limits, subprocess deadline and cleanup
remain unchanged. Sail has no equivalent to Modal's sandbox lifetime limit;
a forcibly killed worker can leave a Sailbox requiring manual termination.
