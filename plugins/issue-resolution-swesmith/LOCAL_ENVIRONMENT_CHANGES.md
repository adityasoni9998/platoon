# Local environment changes

This file records changes made outside Git's tracked source tree while setting
up and debugging the SWE-smith AReaL run. It is a point-in-time inventory from
2026-08-04. None of the paths described below are included in the currently
staged source changes unless explicitly stated.

## AReaL `.venv` hotfix

The installed AReaL package is version `1.0.4`, installed from commit
`67bc932cc2e214ffa6cbd26eb640e16e5dc977b5` of
`https://github.com/inclusionAI/AReaL.git`.

The ignored file

```text
plugins/issue-resolution-swesmith/.venv/lib/python3.12/site-packages/areal/experimental/openai/client.py
```

has two local additions to `GenerationHyperparameters`:

```python
# Chat Completions request path
max_tokens=max_total_tokens_final or 32768,

# Responses API request path
max_tokens=self.engine_max_tokens or 32768,
```

The purpose is to pass the engine's total context limit into AReaL's internal
generation request. `max_new_tokens` continues to bound completion length,
while `max_tokens` carries the prompt-plus-completion limit. With the staged
YAML, the intended total engine limit is 65,000 tokens and the per-call
completion limit is 2,048 tokens.

This is an in-place virtual-environment hotfix. It is not tracked by Git and
can be lost when `.venv` is recreated or when `uv sync` reinstalls AReaL. The
long-term fix should be supplied by a pinned AReaL revision containing the same
upstream change, after which these two local lines should be removed.

At the time of this inventory, this was the only non-`__pycache__` file under
the installed `areal` package modified after the hotfix was applied. A diff
against the checked-out upstream file showed exactly these two added lines.

## User-local tools

The following tools were installed outside this repository:

- Apptainer 1.5.3 at `/home/adityabs/.local/apptainer`. The public launcher is
  `/home/adityabs/.local/apptainer/bin/apptainer`; it adds the bundled utility
  binaries and libraries before executing the architecture-specific binary.
- The user-local Apptainer configuration is
  `/home/adityabs/.local/apptainer/x86_64/etc/apptainer/apptainer.conf`. Notable
  local settings include a 128 MiB session directory, hostfs disabled, user
  binds enabled, and overlay/underlay enabled.
- GNU patch 2.7.6 at `/home/adityabs/.local/opt/gnu-patch/usr/bin/patch` for the
  localization reward's host patch executable.

These files are not represented by staged Git content. The staged `areal.md`
documents the environment variables that point the plugin at these tools.

### How Apptainer was installed without root

This was a relocatable RPM extraction, not a system package installation and
not a source build. No files were installed into `/usr`, no package-manager
database was changed, and no root-owned setuid helper was created.

1. The machine had `cpio` but did not have `rpm2cpio`. To satisfy the official
   installer without root, the following Ubuntu Jammy packages were downloaded
   as `.deb` archives and unpacked into
   `/home/adityabs/.local/opt/rpm-tools` using the already available `ar` and
   archive tools:

   ```text
   rpm2cpio_4.17.0+dfsg1-4build1_amd64.deb
   rpm-common_4.17.0+dfsg1-4build1_amd64.deb
   librpm9_4.17.0+dfsg1-4build1_amd64.deb
   librpmio9_4.17.0+dfsg1-4build1_amd64.deb
   ```

2. A launcher was created at `/home/adityabs/.local/bin/rpm2cpio`. It sets
   `RPM_CONFIGDIR` and `LD_LIBRARY_PATH` to the extracted private RPM runtime,
   then executes
   `/home/adityabs/.local/opt/rpm-tools/usr/bin/rpm2cpio`. This worked around
   the missing system `rpm2cpio` without installing anything globally.

3. Apptainer's official `install-unprivileged.sh` was retained at
   `/tmp/apptainer-installer.hidO0d/install-unprivileged.sh`. With the private
   `rpm2cpio` on `PATH`, it downloaded the EL9 x86-64 Apptainer RPM and its
   runtime RPM dependencies, streamed each through `rpm2cpio | cpio`, and
   assembled a relocatable installation under
   `/home/adityabs/.local/apptainer/x86_64`. The installed build identifies
   itself as `apptainer 1.5.3-1.el9`.

4. The installer also extracted private copies of runtime utilities and
   libraries, including fakeroot, squashfs tools, FUSE libraries, libseccomp,
   and their dependencies, under `x86_64/utils`. It patched the fakeroot helper
   to use paths relative to the installation and generated wrapper scripts so
   the bundled utilities and libraries are selected at runtime.

5. The top-level launcher
   `/home/adityabs/.local/apptainer/bin/apptainer` detects the current
   architecture, prepends `x86_64/utils/bin` to `PATH`, appends
   `x86_64/utils/lib` to `LD_LIBRARY_PATH`, and executes the extracted
   architecture-specific Apptainer binary. Consequently the run environment
   only needs:

   ```bash
   export PATH="/home/adityabs/.local/apptainer/bin:/home/adityabs/.local/bin:$PATH"
   export APPTAINER_EXECUTABLE="/home/adityabs/.local/apptainer/bin/apptainer"
   ```

6. The extracted configuration at
   `/home/adityabs/.local/apptainer/x86_64/etc/apptainer/apptainer.conf` was
   adjusted for these workloads, most notably setting
   `sessiondir max size = 128`. The current observed configuration also has
   `mount hostfs = no`, user bind control enabled, and overlay/underlay enabled.

The build configuration reports `APPTAINER_SUID_INSTALL=0`. Although the text
configuration contains `allow setuid = yes`, this user-owned relocatable build
has no root-owned setuid starter and therefore operates through Apptainer's
unprivileged/user-namespace path. `--fakeroot` is available through that path
and the bundled fakeroot support; it does not grant host root privileges.

## Temporary data and checkouts

The setup and debugging work also created machine-local data under `/tmp`:

- `/tmp/adityabs-apptainer/cache`: Apptainer cache.
- `/tmp/adityabs-apptainer/tmp`: Apptainer temporary workspace.
- `/tmp/adityabs-apptainer/swesmith-build`: SWE-smith SIF artifacts. It
  contained 131 `.sif` files at inventory time.
- `/tmp/platoon_skyrl_areal`: AReaL experiment outputs, rollout JSONL events,
  logs, and name-resolution records.
- `/tmp/benchmarks`: `adityasoni9998/benchmarks`, branch `main`, observed at
  commit `5ea9ebb`.
- `/tmp/platoon-apga-update-areal-registry` and
  `/tmp/platoon-update-areal-registry.vOoUq7`: diagnostic checkouts of
  `ApGa/platoon`, branch `apga/update-areal-with-registry`, observed at commit
  `cf53c71`.
- `/tmp/areal-d99124`, `/tmp/areal-67bc-token-fix`, and
  `/tmp/areal-adityabs`: temporary AReaL source/reference trees used to compare
  behavior and test the context-limit fix.

These are caches, build outputs, logs, or diagnostic clones; none is staged in
the Hyperion repository.

## Git index state at inventory time

All tracked repository modifications were staged, and there were no unstaged
tracked changes. The staged paths were:

```text
platoon/train/areal/patches.py
platoon/train/areal/workflows/group_rollout_workflow.py
plugins/issue-resolution-swesmith/areal.md
plugins/issue-resolution-swesmith/platoon/issue_resolution/localization_reward/localization_patch_processing.py
plugins/issue-resolution-swesmith/platoon/issue_resolution/localization_reward/localization_reward.py
plugins/issue-resolution-swesmith/platoon/issue_resolution/rollout.py
plugins/issue-resolution-swesmith/platoon/issue_resolution/tasks.py
plugins/issue-resolution-swesmith/platoon/issue_resolution/train_areal.py
plugins/issue-resolution-swesmith/platoon/issue_resolution/train_issue_resolution_fft_areal_fsdp.yaml
plugins/issue-resolution-swesmith/pyproject.toml
plugins/issue-resolution-swesmith/uv.lock
```

This Markdown inventory itself was created without changing the existing Git
index.
